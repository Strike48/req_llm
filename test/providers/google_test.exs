defmodule ReqLLM.Providers.GoogleTest do
  @moduledoc """
  Provider-level tests for Google implementation.

  Tests the provider contract directly without going through Generation layer.
  Focus: prepare_request -> attach -> request -> decode pipeline.
  """

  use ReqLLM.ProviderCase, provider: ReqLLM.Providers.Google

  alias ReqLLM.Context
  alias ReqLLM.Providers.Google

  describe "provider contract" do
    test "provider identity and configuration" do
      assert is_atom(Google.provider_id())
      assert is_binary(Google.default_base_url())
      assert String.starts_with?(Google.default_base_url(), "http")
    end

    test "provider schema separation from core options" do
      schema_keys = Google.provider_schema().schema |> Keyword.keys()
      core_keys = ReqLLM.Provider.Options.generation_schema().schema |> Keyword.keys()

      # Provider-specific keys should not overlap with core generation keys
      overlap = MapSet.intersection(MapSet.new(schema_keys), MapSet.new(core_keys))

      assert MapSet.size(overlap) == 0,
             "Schema overlap detected: #{inspect(MapSet.to_list(overlap))}"
    end

    test "supported options include core generation keys" do
      supported = Google.supported_provider_options()
      core_keys = ReqLLM.Provider.Options.all_generation_keys()

      # All core keys should be supported (except meta-keys like :provider_options)
      core_without_meta = Enum.reject(core_keys, &(&1 == :provider_options))
      missing = core_without_meta -- supported
      assert missing == [], "Missing core generation keys: #{inspect(missing)}"
    end

    test "provider_extended_generation_schema includes both base and provider options" do
      extended_schema = Google.provider_extended_generation_schema()
      extended_keys = extended_schema.schema |> Keyword.keys()

      # Should include all core generation keys
      core_keys = ReqLLM.Provider.Options.all_generation_keys()
      core_without_meta = Enum.reject(core_keys, &(&1 == :provider_options))

      for core_key <- core_without_meta do
        assert core_key in extended_keys,
               "Extended schema missing core key: #{core_key}"
      end

      # Should include provider-specific keys
      provider_keys = Google.provider_schema().schema |> Keyword.keys()

      for provider_key <- provider_keys do
        assert provider_key in extended_keys,
               "Extended schema missing provider key: #{provider_key}"
      end
    end
  end

  describe "request preparation & pipeline wiring" do
    test "prepare_request creates configured request" do
      model = ReqLLM.Model.from!("google:gemini-1.5-flash")
      context = context_fixture()
      opts = [temperature: 0.7, max_tokens: 100]

      {:ok, request} = Google.prepare_request(:chat, model, context, opts)

      assert %Req.Request{} = request
      assert request.url.path == "/models/#{model.model}:generateContent"
      assert request.method == :post
    end

    test "prepare_request for streaming creates streaming endpoint" do
      model = ReqLLM.Model.from!("google:gemini-1.5-flash")
      context = context_fixture()
      opts = [temperature: 0.7, max_tokens: 100, stream: true]

      {:ok, request} = Google.prepare_request(:chat, model, context, opts)

      assert %Req.Request{} = request
      assert request.url.path == "/models/#{model.model}:streamGenerateContent"
      assert request.method == :post
    end

    test "attach configures authentication and pipeline" do
      model = ReqLLM.Model.from!("google:gemini-1.5-flash")
      opts = [temperature: 0.5, max_tokens: 50]

      request = Req.new() |> Google.attach(model, opts)

      # Verify core options
      assert request.options[:model] == model.model
      assert request.options[:temperature] == 0.5
      assert request.options[:max_tokens] == 50

      # Google uses query parameter authentication
      assert request.options[:params][:key] != nil

      # Verify pipeline steps
      request_steps = Keyword.keys(request.request_steps)
      response_steps = Keyword.keys(request.response_steps)

      assert :llm_encode_body in request_steps
      assert :llm_decode_response in response_steps
    end

    test "error handling for invalid configurations" do
      model = ReqLLM.Model.from!("google:gemini-1.5-flash")
      context = context_fixture()

      # Unsupported operation
      {:error, error} = Google.prepare_request(:unsupported, model, context, [])
      assert %ReqLLM.Error.Invalid.Parameter{} = error

      # Provider mismatch
      wrong_model = ReqLLM.Model.from!("openai:gpt-4")

      assert_raise ReqLLM.Error.Invalid.Provider, fn ->
        Req.new() |> Google.attach(wrong_model, [])
      end
    end
  end

  describe "body encoding & context translation" do
    test "encode_body without tools" do
      model = ReqLLM.Model.from!("google:gemini-1.5-flash")
      context = context_fixture()

      # Create a mock request with the expected structure
      mock_request = %Req.Request{
        options: [
          context: context,
          model: model.model,
          stream: false
        ]
      }

      # Test the encode_body function directly
      updated_request = Google.encode_body(mock_request)

      assert is_binary(updated_request.body)
      decoded = Jason.decode!(updated_request.body)

      # Should have Google's structure
      assert is_list(decoded["contents"])
      # Only user message, system is separate
      assert length(decoded["contents"]) == 1
      refute Map.has_key?(decoded, "tools")

      # Check generationConfig
      assert Map.has_key?(decoded, "generationConfig")
      assert decoded["generationConfig"]["candidateCount"] == 1

      # Check system instruction is separate
      assert Map.has_key?(decoded, "systemInstruction")
      assert decoded["systemInstruction"]["parts"]

      # Only user message in contents now
      [user_msg] = decoded["contents"]
      assert user_msg["role"] == "user"
      assert is_list(user_msg["parts"])
    end

    test "encode_body with tools but no tool_choice" do
      model = ReqLLM.Model.from!("google:gemini-1.5-flash")
      context = context_fixture()

      tool =
        ReqLLM.Tool.new!(
          name: "test_tool",
          description: "A test tool",
          parameter_schema: [
            name: [type: :string, required: true, doc: "A name parameter"]
          ],
          callback: fn _ -> {:ok, "result"} end
        )

      mock_request = %Req.Request{
        options: [
          context: context,
          model: model.model,
          stream: false,
          tools: [tool],
          operation: :chat
        ]
      }

      updated_request = Google.encode_body(mock_request)
      decoded = Jason.decode!(updated_request.body)

      assert is_list(decoded["tools"])
      assert length(decoded["tools"]) == 1

      [tool_def] = decoded["tools"]
      assert Map.has_key?(tool_def, "functionDeclarations")
      assert is_list(tool_def["functionDeclarations"])
    end

    test "encode_body with Google-specific options" do
      model = ReqLLM.Model.from!("google:gemini-1.5-flash")
      context = context_fixture()

      safety_settings = [
        %{
          "category" => "HARM_CATEGORY_HARASSMENT",
          "threshold" => "BLOCK_MEDIUM_AND_ABOVE"
        }
      ]

      mock_request = %Req.Request{
        options: [
          context: context,
          model: model.model,
          stream: false,
          google_safety_settings: safety_settings,
          google_candidate_count: 2,
          temperature: 0.8,
          max_tokens: 500
        ]
      }

      updated_request = Google.encode_body(mock_request)
      decoded = Jason.decode!(updated_request.body)

      # Check safety settings
      assert decoded["safetySettings"] == safety_settings

      # Check generation config with Google parameter names
      gen_config = decoded["generationConfig"]
      assert gen_config["temperature"] == 0.8
      assert gen_config["maxOutputTokens"] == 500
      assert gen_config["candidateCount"] == 2
    end

    test "encode_body for embedding operation" do
      mock_request = %Req.Request{
        options: [
          operation: :embedding,
          model: "gemini-embedding-001",
          text: "Hello, world!",
          dimensions: 768
        ]
      }

      updated_request = Google.encode_body(mock_request)
      decoded = Jason.decode!(updated_request.body)

      # Check embedding-specific structure
      assert decoded["model"] == "models/gemini-embedding-001"
      assert decoded["content"]["parts"] == [%{"text" => "Hello, world!"}]
      assert decoded["outputDimensionality"] == 768
    end
  end

  describe "response decoding" do
    test "decode_response handles non-streaming responses" do
      # Create a mock Google response
      google_response = %{
        "candidates" => [
          %{
            "content" => %{
              "parts" => [%{"text" => "Hello there! How can I help you today?"}],
              "role" => "model"
            },
            "finishReason" => "STOP",
            "safetyRatings" => []
          }
        ],
        "usageMetadata" => %{
          "promptTokenCount" => 10,
          "candidatesTokenCount" => 15,
          "totalTokenCount" => 25
        }
      }

      # Create a mock Req response
      mock_resp = %Req.Response{
        status: 200,
        body: google_response
      }

      model = ReqLLM.Model.from!("google:gemini-1.5-flash")
      context = context_fixture()

      mock_req = %Req.Request{
        options: [context: context, stream: false, model: model.model]
      }

      # Test decode_response directly
      {req, resp} = Google.decode_response({mock_req, mock_resp})

      assert req == mock_req
      assert %ReqLLM.Response{} = resp.body

      response = resp.body
      assert is_binary(response.id)
      assert response.model == model.model
      assert response.stream? == false

      # Verify message normalization
      assert response.message.role == :assistant
      text = ReqLLM.Response.text(response)
      assert text == "Hello there! How can I help you today?"
      assert response.finish_reason == :stop

      # Verify usage normalization
      assert response.usage.input_tokens == 10
      assert response.usage.output_tokens == 15
      assert response.usage.total_tokens == 25

      # Verify context advancement (original + assistant)
      assert length(response.context.messages) == 3
      assert List.last(response.context.messages).role == :assistant
    end

    test "decode_response preserves tool calls" do
      google_response = %{
        "candidates" => [
          %{
            "content" => %{
              "parts" => [
                %{
                  "functionCall" => %{
                    "name" => "demo_tool",
                    "args" => %{"payload" => "value"},
                    "id" => "call-1"
                  }
                }
              ],
              "role" => "model"
            },
            "finishReason" => "STOP"
          }
        ],
        "usageMetadata" => %{
          "promptTokenCount" => 12,
          "candidatesTokenCount" => 3,
          "totalTokenCount" => 15
        }
      }

      mock_resp = %Req.Response{status: 200, body: google_response}

      context = context_fixture()

      mock_req = %Req.Request{
        options: [context: context, stream: false, model: "gemini-1.5-flash"]
      }

      {_req, resp} = Google.decode_response({mock_req, mock_resp})

      response = resp.body
      assert %ReqLLM.Response{} = response

      assert [tool_call] = ReqLLM.Response.tool_calls(response)
      assert tool_call.function.name == "demo_tool"
      assert tool_call.id == "call-1"
      assert ReqLLM.Response.finish_reason(response) == :stop
    end

    test "decode_response handles streaming responses" do
      # Create mock streaming chunks (Google format)
      stream_chunks = [
        %{"candidates" => [%{"content" => %{"parts" => [%{"text" => "Hello"}]}}]},
        %{"candidates" => [%{"content" => %{"parts" => [%{"text" => " world"}]}}]},
        %{"candidates" => [%{"finishReason" => "STOP"}]}
      ]

      # Create a mock stream
      mock_stream = Stream.map(stream_chunks, & &1)

      # Create a mock Req response with streaming body
      mock_resp = %Req.Response{
        status: 200,
        body: mock_stream
      }

      # Create a mock request with context and model
      context = context_fixture()
      model = "gemini-1.5-flash"

      mock_req = %Req.Request{
        options: [context: context, stream: true, model: model]
      }

      # Test decode_response directly
      {req, resp} = Google.decode_response({mock_req, mock_resp})

      assert req == mock_req
      assert %ReqLLM.Response{} = resp.body

      response = resp.body
      assert response.stream? == true
      assert is_struct(response.stream, Stream)
      assert response.model == model

      # Verify context is preserved (original messages only in streaming)
      assert length(response.context.messages) == 2

      # Verify stream structure and processing
      assert response.usage == %{
               input_tokens: 0,
               output_tokens: 0,
               total_tokens: 0,
               cached_tokens: 0,
               reasoning_tokens: 0
             }

      assert response.finish_reason == nil
      assert response.provider_meta == %{}
    end

    test "decode_response for embedding returns normalized OpenAI format" do
      embedding_response = %{
        "embedding" => %{
          "values" => [0.1, -0.2, 0.3, 0.4, -0.5]
        }
      }

      mock_resp = %Req.Response{
        status: 200,
        body: embedding_response
      }

      mock_req = %Req.Request{
        options: [operation: :embedding, model: "gemini-embedding-001"]
      }

      {req, resp} = Google.decode_response({mock_req, mock_resp})

      assert req == mock_req

      assert resp.body == %{
               "data" => [%{"index" => 0, "embedding" => [0.1, -0.2, 0.3, 0.4, -0.5]}]
             }
    end

    test "decode_response handles API errors with non-200 status" do
      # Create error response
      error_body = %{
        "error" => %{
          "code" => 400,
          "message" => "Invalid API key",
          "status" => "INVALID_ARGUMENT"
        }
      }

      mock_resp = %Req.Response{
        status: 400,
        body: error_body
      }

      context = context_fixture()

      mock_req = %Req.Request{
        options: [context: context, model: "gemini-1.5-flash"]
      }

      # Test decode_response error handling
      {req, error} = Google.decode_response({mock_req, mock_resp})

      assert req == mock_req
      assert %ReqLLM.Error.API.Response{} = error
      assert error.status == 400
      assert error.reason == "Google API error"
      assert error.response_body == error_body
    end
  end

  describe "option translation" do
    test "provider implements translate_options/3" do
      # Google implements translate_options/3 for stream? alias handling
      assert function_exported?(Google, :translate_options, 3)
    end

    test "translate_options handles stream? alias" do
      model = ReqLLM.Model.from!("google:gemini-1.5-flash")

      # Test stream? -> stream translation
      opts = [temperature: 0.7, stream?: true]
      {translated_opts, warnings} = Google.translate_options(:chat, model, opts)

      assert Keyword.get(translated_opts, :stream) == true
      refute Keyword.has_key?(translated_opts, :stream?)
      assert warnings == []
    end

    test "provider-specific option handling" do
      # Test that provider-specific options are present in the provider schema
      schema_keys = Google.provider_schema().schema |> Keyword.keys()

      # Test that these options are supported
      supported_opts = Google.supported_provider_options()

      for provider_option <- schema_keys do
        assert provider_option in supported_opts,
               "Expected #{provider_option} to be in supported options"
      end
    end
  end

  describe "usage extraction" do
    test "extract_usage with valid usage data" do
      model = ReqLLM.Model.from!("google:gemini-1.5-flash")

      body_with_usage = %{
        "usageMetadata" => %{
          "promptTokenCount" => 10,
          "candidatesTokenCount" => 20,
          "totalTokenCount" => 30
        }
      }

      {:ok, usage} = Google.extract_usage(body_with_usage, model)
      assert usage.input_tokens == 10
      assert usage.output_tokens == 20
      assert usage.total_tokens == 30
    end

    test "extract_usage with missing usage data" do
      model = ReqLLM.Model.from!("google:gemini-1.5-flash")
      body_without_usage = %{"candidates" => []}

      {:error, :no_usage_found} = Google.extract_usage(body_without_usage, model)
    end

    test "extract_usage with invalid body type" do
      model = ReqLLM.Model.from!("google:gemini-1.5-flash")

      {:error, :invalid_body} = Google.extract_usage("invalid", model)
      {:error, :invalid_body} = Google.extract_usage(nil, model)
      {:error, :invalid_body} = Google.extract_usage(123, model)
    end

    test "extract_usage with cached content tokens" do
      model = ReqLLM.Model.from!("google:gemini-1.5-flash")

      body_with_cached_tokens = %{
        "usageMetadata" => %{
          "promptTokenCount" => 500,
          "candidatesTokenCount" => 200,
          "totalTokenCount" => 700,
          "cachedContentTokenCount" => 120
        }
      }

      {:ok, usage} = Google.extract_usage(body_with_cached_tokens, model)
      assert usage.input_tokens == 500
      assert usage.output_tokens == 200
      assert usage.total_tokens == 700
      assert usage.cached_tokens == 120
    end

    test "extract_usage without cached content tokens" do
      model = ReqLLM.Model.from!("google:gemini-1.5-flash")

      body_without_cached_tokens = %{
        "usageMetadata" => %{
          "promptTokenCount" => 10,
          "candidatesTokenCount" => 20,
          "totalTokenCount" => 30
        }
      }

      {:ok, usage} = Google.extract_usage(body_without_cached_tokens, model)
      assert usage.input_tokens == 10
      assert usage.output_tokens == 20
      assert usage.total_tokens == 30
      assert usage.cached_tokens == 0
    end
  end

  describe "object generation with native JSON mode" do
    test "prepare_request for :object with low max_tokens gets adjusted" do
      model = ReqLLM.Model.from!("google:gemini-1.5-flash")
      context = context_fixture()
      {:ok, schema} = ReqLLM.Schema.compile(name: [type: :string, required: true])

      opts = [max_tokens: 50, compiled_schema: schema]
      {:ok, request} = Google.prepare_request(:object, model, context, opts)

      assert request.options[:max_tokens] == 200
    end

    test "prepare_request for :object with nil max_tokens gets default" do
      model = ReqLLM.Model.from!("google:gemini-1.5-flash")
      context = context_fixture()
      {:ok, schema} = ReqLLM.Schema.compile([])

      opts = [compiled_schema: schema]
      {:ok, request} = Google.prepare_request(:object, model, context, opts)

      assert request.options[:max_tokens] == 4096
    end

    test "prepare_request for :object with sufficient max_tokens unchanged" do
      model = ReqLLM.Model.from!("google:gemini-1.5-flash")
      context = context_fixture()
      {:ok, schema} = ReqLLM.Schema.compile(value: [type: :integer])

      opts = [max_tokens: 1000, compiled_schema: schema]
      {:ok, request} = Google.prepare_request(:object, model, context, opts)

      assert request.options[:max_tokens] == 1000
    end

    test "prepare_request for :object rejects tools" do
      model = ReqLLM.Model.from!("google:gemini-1.5-flash")
      context = context_fixture()
      {:ok, schema} = ReqLLM.Schema.compile(name: [type: :string])

      tool =
        ReqLLM.Tool.new!(
          name: "test_tool",
          description: "A test",
          parameter_schema: [],
          callback: fn _ -> {:ok, "ok"} end
        )

      opts = [compiled_schema: schema, tools: [tool]]
      {:error, error} = Google.prepare_request(:object, model, context, opts)

      assert %ReqLLM.Error.Invalid.Parameter{} = error
      assert error.parameter =~ "tools are not supported"
    end

    test "encode_object_body creates JSON mode request" do
      model = ReqLLM.Model.from!("google:gemini-1.5-flash")
      context = context_fixture()
      {:ok, schema} = ReqLLM.Schema.compile(name: [type: :string, required: true])

      mock_request = %Req.Request{
        options: [
          context: context,
          model: model.model,
          operation: :object,
          compiled_schema: schema,
          max_tokens: 500
        ]
      }

      updated_request = Google.encode_body(mock_request)
      decoded = Jason.decode!(updated_request.body)

      assert decoded["generationConfig"]["responseMimeType"] == "application/json"
      assert decoded["generationConfig"]["candidateCount"] == 1
      assert Map.has_key?(decoded["generationConfig"], "responseSchema")
      refute Map.has_key?(decoded, "tools")
      refute Map.has_key?(decoded, "toolConfig")
    end

    test "encode_object_body uses responseSchema for non-2.5 models" do
      context = context_fixture()

      {:ok, schema} =
        ReqLLM.Schema.compile(
          name: [type: :string, required: true],
          age: [type: :pos_integer]
        )

      mock_request = %Req.Request{
        options: [
          context: context,
          model: "gemini-1.5-flash",
          operation: :object,
          compiled_schema: schema
        ]
      }

      updated_request = Google.encode_body(mock_request)
      decoded = Jason.decode!(updated_request.body)

      response_schema = decoded["generationConfig"]["responseSchema"]
      assert response_schema["type"] == "OBJECT"
      assert Map.has_key?(response_schema, "properties")
      assert Map.has_key?(response_schema["properties"], "name")
      assert response_schema["properties"]["name"]["type"] == "STRING"
      refute Map.has_key?(decoded["generationConfig"], "responseJsonSchema")
    end

    test "encode_object_body uses responseJsonSchema for Gemini 2.5" do
      context = context_fixture()

      {:ok, schema} =
        ReqLLM.Schema.compile(
          name: [type: :string, required: true],
          age: [type: :pos_integer]
        )

      mock_request = %Req.Request{
        options: [
          context: context,
          model: "gemini-2.5-flash",
          operation: :object,
          compiled_schema: schema
        ]
      }

      updated_request = Google.encode_body(mock_request)
      decoded = Jason.decode!(updated_request.body)

      response_json_schema = decoded["generationConfig"]["responseJsonSchema"]
      assert response_json_schema["type"] == "object"
      assert Map.has_key?(response_json_schema, "properties")
      refute Map.has_key?(decoded["generationConfig"], "responseSchema")
    end

    test "prepare_request creates configured embedding request" do
      model = ReqLLM.Model.from!("google:gemini-embedding-001")
      text = "Hello, world!"
      opts = [dimensions: 768]

      {:ok, request} = Google.prepare_request(:embedding, model, text, opts)

      assert request.method == :post
      assert request.url.path == "/models/gemini-embedding-001:embedContent"

      # Check request options contain embedding-specific data
      assert request.options[:text] == text
      assert request.options[:operation] == :embedding
      assert request.options[:dimensions] == 768
    end

    test "prepare_request rejects unsupported operations" do
      model = ReqLLM.Model.from!("google:gemini-1.5-flash")
      context = context_fixture()

      # Test unsupported operation for 3-arg version
      {:error, error} = Google.prepare_request(:unsupported_operation, model, context, [])
      assert %ReqLLM.Error.Invalid.Parameter{} = error
      assert error.parameter =~ "operation: :unsupported_operation not supported"

      # Test unsupported operation for object with schema
      {:ok, schema} = ReqLLM.Schema.compile([])

      {:error, error} =
        Google.prepare_request(:unsupported_operation, model, context, compiled_schema: schema)

      assert %ReqLLM.Error.Invalid.Parameter{} = error
      assert error.parameter =~ "operation: :unsupported_operation not supported"
    end
  end

  describe "tool result encoding (agentic loops)" do
    test "encode_body with Context containing tool result messages" do
      # Create a multi-turn conversation with tool calls and results
      context =
        Context.new([
          Context.system("You are a helpful assistant"),
          Context.user("What's the weather?"),
          Context.assistant("Let me check.",
            tool_calls: [
              %{
                id: "call_123",
                name: "get_weather",
                arguments: %{"location" => "SF"}
              }
            ]
          ),
          Context.tool_result_message("get_weather", "call_123", %{
            "temperature" => 72,
            "conditions" => "sunny"
          })
        ])

      mock_request = %Req.Request{
        options: [
          context: context,
          model: "gemini-1.5-flash",
          stream: false
        ]
      }

      updated_request = Google.encode_body(mock_request)
      decoded = Jason.decode!(updated_request.body)

      # Verify structure
      assert Map.has_key?(decoded, "systemInstruction")
      contents = decoded["contents"]
      assert length(contents) == 3

      # User message
      assert Enum.at(contents, 0)["role"] == "user"
      assert Enum.at(contents, 0)["parts"] == [%{"text" => "What's the weather?"}]

      # Assistant message with tool call
      assert Enum.at(contents, 1)["role"] == "model"
      parts_1 = Enum.at(contents, 1)["parts"]
      assert length(parts_1) == 2
      assert Enum.at(parts_1, 0) == %{"text" => "Let me check."}

      assert Enum.at(parts_1, 1) == %{
               "functionCall" => %{
                 "name" => "get_weather",
                 "args" => %{"location" => "SF"}
               }
             }

      # Tool result message (as user role with functionResponse)
      assert Enum.at(contents, 2)["role"] == "user"
      tool_result_parts = Enum.at(contents, 2)["parts"]
      assert length(tool_result_parts) == 1

      assert tool_result_parts == [
               %{
                 "functionResponse" => %{
                   "name" => "get_weather",
                   "response" => %{
                     "temperature" => 72,
                     "conditions" => "sunny"
                   }
                 }
               }
             ]
    end

    test "encode_body handles multiple tool results in sequence" do
      # Simulate a conversation with multiple tool calls
      context =
        Context.new([
          Context.user("Check weather in SF and NYC"),
          Context.assistant("",
            tool_calls: [
              %{id: "call_1", name: "get_weather", arguments: %{"location" => "SF"}},
              %{id: "call_2", name: "get_weather", arguments: %{"location" => "NYC"}}
            ]
          ),
          Context.tool_result_message("get_weather", "call_1", %{"temp" => 72}),
          Context.tool_result_message("get_weather", "call_2", %{"temp" => 65})
        ])

      mock_request = %Req.Request{
        options: [
          context: context,
          model: "gemini-1.5-flash"
        ]
      }

      updated_request = Google.encode_body(mock_request)
      decoded = Jason.decode!(updated_request.body)

      contents = decoded["contents"]
      assert length(contents) == 4

      # Assistant with two tool calls
      model_msg = Enum.at(contents, 1)
      assert model_msg["role"] == "model"
      assert length(model_msg["parts"]) == 2

      # Two separate tool result messages
      tool_result_1 = Enum.at(contents, 2)
      assert tool_result_1["role"] == "user"

      assert tool_result_1["parts"] == [
               %{
                 "functionResponse" => %{
                   "name" => "get_weather",
                   "response" => %{"temp" => 72}
                 }
               }
             ]

      tool_result_2 = Enum.at(contents, 3)
      assert tool_result_2["role"] == "user"

      assert tool_result_2["parts"] == [
               %{
                 "functionResponse" => %{
                   "name" => "get_weather",
                   "response" => %{"temp" => 65}
                 }
               }
             ]
    end

    test "encode_body with tool calls and results using Context API" do
      # Create a conversation using proper Context API
      context =
        Context.new([
          Context.user("Test"),
          Context.assistant("Calling tool",
            tool_calls: [
              %{id: "tool_123", name: "test_tool", arguments: %{"param" => "value"}}
            ]
          ),
          Context.tool_result_message("test_tool", "tool_123", %{"result" => "success"})
        ])

      mock_request = %Req.Request{
        options: [
          context: context,
          model: "gemini-1.5-flash"
        ]
      }

      updated_request = Google.encode_body(mock_request)
      decoded = Jason.decode!(updated_request.body)

      contents = decoded["contents"]
      assert length(contents) == 3

      # Check tool result is properly formatted
      tool_result_msg = Enum.at(contents, 2)
      assert tool_result_msg["role"] == "user"

      assert tool_result_msg["parts"] == [
               %{
                 "functionResponse" => %{
                   "name" => "test_tool",
                   "response" => %{"result" => "success"}
                 }
               }
             ]
    end

    test "encode_body preserves tool name from message.name field" do
      # Create tool result message using Context API
      context =
        Context.new([
          Context.tool_result_message("my_custom_tool", "call_xyz", %{"data" => "test"})
        ])

      mock_request = %Req.Request{
        options: [
          context: context,
          model: "gemini-1.5-flash"
        ]
      }

      updated_request = Google.encode_body(mock_request)
      decoded = Jason.decode!(updated_request.body)

      [tool_msg] = decoded["contents"]

      assert tool_msg["parts"] == [
               %{
                 "functionResponse" => %{
                   "name" => "my_custom_tool",
                   "response" => %{"data" => "test"}
                 }
               }
             ]
    end

    test "encode_body handles empty tool result" do
      context =
        Context.new([
          Context.user("Test"),
          Context.assistant("",
            tool_calls: [
              %{id: "call_1", name: "void_tool", arguments: %{}}
            ]
          ),
          Context.tool_result_message("void_tool", "call_1", %{})
        ])

      mock_request = %Req.Request{
        options: [
          context: context,
          model: "gemini-1.5-flash"
        ]
      }

      updated_request = Google.encode_body(mock_request)
      decoded = Jason.decode!(updated_request.body)

      contents = decoded["contents"]
      tool_result = Enum.at(contents, 2)

      assert tool_result["parts"] == [
               %{
                 "functionResponse" => %{
                   "name" => "void_tool",
                   "response" => %{}
                 }
               }
             ]
    end

    test "encode_body handles complex nested tool result data" do
      complex_result = %{
        "status" => "success",
        "data" => %{
          "items" => [
            %{"id" => 1, "name" => "Item 1"},
            %{"id" => 2, "name" => "Item 2"}
          ],
          "metadata" => %{
            "count" => 2,
            "timestamp" => "2024-01-01T00:00:00Z"
          }
        }
      }

      context =
        Context.new([
          Context.user("Fetch items"),
          Context.assistant("",
            tool_calls: [
              %{id: "call_1", name: "fetch_items", arguments: %{}}
            ]
          ),
          Context.tool_result_message("fetch_items", "call_1", complex_result)
        ])

      mock_request = %Req.Request{
        options: [
          context: context,
          model: "gemini-1.5-flash"
        ]
      }

      updated_request = Google.encode_body(mock_request)
      decoded = Jason.decode!(updated_request.body)

      contents = decoded["contents"]
      tool_result = Enum.at(contents, 2)

      assert tool_result["parts"] == [
               %{
                 "functionResponse" => %{
                   "name" => "fetch_items",
                   "response" => complex_result
                 }
               }
             ]
    end

    test "full agentic loop: user -> tool call -> tool result -> final response" do
      # Simulate a complete agentic loop
      initial_context =
        Context.new([
          Context.system("You are a helpful assistant"),
          Context.user("What's 2+2?")
        ])

      # First request - LLM calls tool
      mock_request_1 = %Req.Request{
        options: [
          context: initial_context,
          model: "gemini-1.5-flash",
          tools: [
            ReqLLM.Tool.new!(
              name: "calculate",
              description: "Perform calculation",
              parameter_schema: [expr: [type: :string, required: true]],
              callback: fn _ -> {:ok, "4"} end
            )
          ]
        ]
      }

      updated_request_1 = Google.encode_body(mock_request_1)
      decoded_1 = Jason.decode!(updated_request_1.body)

      # Verify tools are sent
      assert Map.has_key?(decoded_1, "tools")

      assert decoded_1["contents"] == [
               %{"role" => "user", "parts" => [%{"text" => "What's 2+2?"}]}
             ]

      # Simulate LLM response with tool call
      # In real scenario, this would come from Google API
      # Then we'd build a new context with the tool result

      context_with_tool_result =
        Context.new([
          Context.system("You are a helpful assistant"),
          Context.user("What's 2+2?"),
          Context.assistant("Let me calculate that.",
            tool_calls: [
              %{id: "call_1", name: "calculate", arguments: %{"expr" => "2+2"}}
            ]
          ),
          Context.tool_result_message("calculate", "call_1", "4")
        ])

      # Second request - Send tool result back
      mock_request_2 = %Req.Request{
        options: [
          context: context_with_tool_result,
          model: "gemini-1.5-flash",
          tools: [
            ReqLLM.Tool.new!(
              name: "calculate",
              description: "Perform calculation",
              parameter_schema: [expr: [type: :string, required: true]],
              callback: fn _ -> {:ok, "4"} end
            )
          ]
        ]
      }

      updated_request_2 = Google.encode_body(mock_request_2)
      decoded_2 = Jason.decode!(updated_request_2.body)

      # Verify the complete conversation is sent
      # System message should be in systemInstruction, not contents
      assert Map.has_key?(decoded_2, "systemInstruction")

      contents = decoded_2["contents"]
      # contents: user, model with tool call, tool result (3 messages)
      assert length(contents) == 3

      # User question
      assert Enum.at(contents, 0)["role"] == "user"

      # Assistant with tool call
      assert Enum.at(contents, 1)["role"] == "model"
      model_parts = Enum.at(contents, 1)["parts"]
      assert Enum.any?(model_parts, &Map.has_key?(&1, "functionCall"))

      # Tool result (third message, not fourth - system is separate)
      assert Enum.at(contents, 2)["role"] == "user"

      assert Enum.at(contents, 2)["parts"] == [
               %{
                 "functionResponse" => %{
                   "name" => "calculate",
                   "response" => "4"
                 }
               }
             ]

      # This allows the LLM to now provide a final answer
    end
  end

  describe "error handling & robustness" do
    test "context validation" do
      # Multiple system messages should fail
      invalid_context =
        Context.new([
          Context.system("System 1"),
          Context.system("System 2"),
          Context.user("Hello")
        ])

      assert_raise ReqLLM.Error.Validation.Error,
                   ~r/should have at most one system message/,
                   fn ->
                     Context.validate!(invalid_context)
                   end
    end
  end

  describe "file attachment support" do
    test "encode_body handles :file ContentPart with inline_data format" do
      file_content = "test file content"

      file_part = %ReqLLM.Message.ContentPart{
        type: :file,
        data: file_content,
        media_type: "application/pdf"
      }

      message_with_file = %ReqLLM.Message{
        role: :user,
        content: [file_part]
      }

      context = %ReqLLM.Context{messages: [message_with_file]}

      mock_request = %Req.Request{
        options: [
          context: context,
          model: "gemini-1.5-flash",
          stream: false
        ]
      }

      updated_request = Google.encode_body(mock_request)
      decoded = Jason.decode!(updated_request.body)

      [user_msg] = decoded["contents"]
      parts = user_msg["parts"]

      assert length(parts) == 1, "Expected 1 part, got: #{inspect(parts)}"
      [part] = parts

      assert Map.has_key?(part, "inline_data")
      assert part["inline_data"]["mime_type"] == "application/pdf"
      assert Base.decode64!(part["inline_data"]["data"]) == file_content
    end

    test "encode_body handles video ContentPart with inline_data format" do
      video_content = "fake video bytes"

      video_part = %ReqLLM.Message.ContentPart{
        type: :file,
        data: video_content,
        media_type: "video/mp4"
      }

      message_with_video = %ReqLLM.Message{
        role: :user,
        content: [video_part]
      }

      context = %ReqLLM.Context{messages: [message_with_video]}

      mock_request = %Req.Request{
        options: [
          context: context,
          model: "gemini-1.5-flash",
          stream: false
        ]
      }

      updated_request = Google.encode_body(mock_request)
      decoded = Jason.decode!(updated_request.body)

      [user_msg] = decoded["contents"]
      [part] = user_msg["parts"]

      assert Map.has_key?(part, "inline_data")
      assert part["inline_data"]["mime_type"] == "video/mp4"
      assert Base.decode64!(part["inline_data"]["data"]) == video_content
    end

    test "encode_body handles image ContentPart with inline_data format" do
      image_content = <<137, 80, 78, 71, 13, 10, 26, 10>>

      image_part = %ReqLLM.Message.ContentPart{
        type: :image,
        data: image_content,
        media_type: "image/png"
      }

      message_with_image = %ReqLLM.Message{
        role: :user,
        content: [image_part]
      }

      context = %ReqLLM.Context{messages: [message_with_image]}

      mock_request = %Req.Request{
        options: [
          context: context,
          model: "gemini-1.5-flash",
          stream: false
        ]
      }

      updated_request = Google.encode_body(mock_request)
      decoded = Jason.decode!(updated_request.body)

      [user_msg] = decoded["contents"]
      [part] = user_msg["parts"]

      assert Map.has_key?(part, "inline_data")
      assert part["inline_data"]["mime_type"] == "image/png"
      assert Base.decode64!(part["inline_data"]["data"]) == image_content
    end
  end
end
