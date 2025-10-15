defmodule ReqLLM.Providers.AnthropicTest do
  @moduledoc """
  Provider-level tests for Anthropic implementation.

  Tests the provider contract directly without going through Generation layer.
  Focus: prepare_request -> attach -> request -> decode pipeline.
  """

  use ReqLLM.ProviderCase, provider: ReqLLM.Providers.Anthropic

  alias ReqLLM.Providers.Anthropic

  describe "provider contract" do
    test "provider identity and configuration" do
      assert is_atom(Anthropic.provider_id())
      assert is_binary(Anthropic.default_base_url())
      assert String.starts_with?(Anthropic.default_base_url(), "http")
    end

    test "provider schema separation from core options" do
      schema_keys = Anthropic.provider_schema().schema |> Keyword.keys()
      core_keys = ReqLLM.Provider.Options.generation_schema().schema |> Keyword.keys()

      # Provider-specific keys should not overlap with core generation keys
      overlap = MapSet.intersection(MapSet.new(schema_keys), MapSet.new(core_keys))

      assert MapSet.size(overlap) == 0,
             "Schema overlap detected: #{inspect(MapSet.to_list(overlap))}"
    end

    test "supported options include core generation keys" do
      supported = Anthropic.supported_provider_options()
      core_keys = ReqLLM.Provider.Options.all_generation_keys()

      # All core keys should be supported (except meta-keys like :provider_options)
      core_without_meta = Enum.reject(core_keys, &(&1 == :provider_options))
      missing = core_without_meta -- supported
      assert missing == [], "Missing core generation keys: #{inspect(missing)}"
    end
  end

  describe "request preparation & pipeline wiring" do
    test "prepare_request creates configured request" do
      model = ReqLLM.Model.from!("anthropic:claude-3-5-sonnet-20241022")
      prompt = "Hello world"
      opts = [temperature: 0.7, max_tokens: 100]

      {:ok, request} = Anthropic.prepare_request(:chat, model, prompt, opts)

      assert %Req.Request{} = request
      assert request.url.path == "/v1/messages"
      assert request.method == :post
    end

    test "attach configures authentication and pipeline" do
      model = ReqLLM.Model.from!("anthropic:claude-3-5-sonnet-20241022")
      opts = [temperature: 0.5, max_tokens: 50]

      request = Req.new() |> Anthropic.attach(model, opts)

      # Verify authentication
      api_key_header = Enum.find(request.headers, fn {name, _} -> name == "x-api-key" end)
      assert api_key_header != nil

      version_header = Enum.find(request.headers, fn {name, _} -> name == "anthropic-version" end)
      assert version_header != nil

      # Verify pipeline steps
      request_steps = Keyword.keys(request.request_steps)
      response_steps = Keyword.keys(request.response_steps)

      assert :llm_encode_body in request_steps
      assert :llm_decode_response in response_steps
    end

    test "error handling for invalid configurations" do
      model = ReqLLM.Model.from!("anthropic:claude-3-5-sonnet-20241022")
      prompt = "Hello world"

      # Unsupported operation
      {:error, error} = Anthropic.prepare_request(:unsupported, model, prompt, [])
      assert %ReqLLM.Error.Invalid.Parameter{} = error

      # Provider mismatch
      wrong_model = ReqLLM.Model.from!("openai:gpt-4")

      assert_raise ReqLLM.Error.Invalid.Provider, fn ->
        Req.new() |> Anthropic.attach(wrong_model, [])
      end
    end
  end

  describe "body encoding & context translation" do
    test "encode_body without tools" do
      model = ReqLLM.Model.from!("anthropic:claude-3-5-sonnet-20241022")
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
      updated_request = Anthropic.encode_body(mock_request)

      assert is_binary(updated_request.body)
      decoded = Jason.decode!(updated_request.body)

      assert decoded["model"] == "claude-3-5-sonnet-20241022"
      assert is_list(decoded["messages"])
      # Only user message, system goes to top-level
      assert length(decoded["messages"]) == 1
      assert decoded["stream"] == false
      refute Map.has_key?(decoded, "tools")

      # Check top-level system parameter (Anthropic format)
      assert decoded["system"] == "You are a helpful assistant."

      [user_msg] = decoded["messages"]
      assert user_msg["role"] == "user"
      assert user_msg["content"] == "Hello, how are you?"
    end

    test "encode_body with tools" do
      model = ReqLLM.Model.from!("anthropic:claude-3-5-sonnet-20241022")
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

      tool_choice = %{type: "tool", name: "test_tool"}

      mock_request = %Req.Request{
        options: [
          context: context,
          model: model.model,
          stream: false,
          tools: [tool],
          tool_choice: tool_choice
        ]
      }

      updated_request = Anthropic.encode_body(mock_request)
      decoded = Jason.decode!(updated_request.body)

      assert is_list(decoded["tools"])
      assert length(decoded["tools"]) == 1
      assert decoded["tool_choice"] == %{"type" => "tool", "name" => "test_tool"}

      [encoded_tool] = decoded["tools"]
      assert encoded_tool["name"] == "test_tool"
      assert encoded_tool["description"] == "A test tool"
      assert is_map(encoded_tool["input_schema"])
    end
  end

  describe "response decoding & normalization" do
    test "decode_response handles non-streaming responses" do
      # Create a mock Anthropic-format response
      mock_json_response = anthropic_format_json_fixture()

      # Create a mock Req response
      mock_resp = %Req.Response{
        status: 200,
        body: mock_json_response
      }

      # Create a mock request with context
      model = ReqLLM.Model.from!("anthropic:claude-3-5-sonnet-20241022")
      context = context_fixture()

      mock_req = %Req.Request{
        options: [context: context, stream: false, model: "anthropic:claude-3-5-sonnet-20241022"]
      }

      # Test decode_response directly
      {req, resp} = Anthropic.decode_response({mock_req, mock_resp})

      assert req == mock_req
      assert %ReqLLM.Response{} = resp.body

      response = resp.body
      assert is_binary(response.id)
      assert response.model == model.model
      assert response.stream? == false

      # Verify message normalization
      assert response.message.role == :assistant
      text = ReqLLM.Response.text(response)
      assert is_binary(text)
      assert String.length(text) > 0
      assert response.finish_reason in [:stop, :length]

      # Verify usage normalization
      assert is_integer(response.usage.input_tokens)
      assert is_integer(response.usage.output_tokens)
      assert is_integer(response.usage.total_tokens)

      # Verify context advancement (original + assistant)
      assert length(response.context.messages) == 3
      assert List.last(response.context.messages).role == :assistant
    end

    test "decode_response handles API errors with non-200 status" do
      # Create error response
      error_body = %{
        "type" => "error",
        "error" => %{
          "type" => "authentication_error",
          "message" => "Invalid API key"
        }
      }

      mock_resp = %Req.Response{
        status: 401,
        body: error_body
      }

      context = context_fixture()

      mock_req = %Req.Request{
        options: [context: context, model: "claude-3-5-sonnet-20241022"]
      }

      # Test decode_response error handling
      {req, error} = Anthropic.decode_response({mock_req, mock_resp})

      assert req == mock_req
      assert %ReqLLM.Error.API.Response{} = error
      assert error.status == 401
      assert error.reason =~ "Anthropic API error"
      assert error.response_body == error_body
    end
  end

  describe "option translation" do
    test "translate_options converts stop to stop_sequences" do
      model = ReqLLM.Model.from!("anthropic:claude-3-5-sonnet-20241022")

      # Test single stop string
      {translated_opts, []} = Anthropic.translate_options(:chat, model, stop: "STOP")
      assert Keyword.get(translated_opts, :stop_sequences) == ["STOP"]
      assert Keyword.get(translated_opts, :stop) == nil

      # Test stop list
      {translated_opts, []} = Anthropic.translate_options(:chat, model, stop: ["STOP", "END"])
      assert Keyword.get(translated_opts, :stop_sequences) == ["STOP", "END"]
      assert Keyword.get(translated_opts, :stop) == nil
    end

    test "translate_options removes unsupported parameters" do
      model = ReqLLM.Model.from!("anthropic:claude-3-5-sonnet-20241022")

      opts = [
        temperature: 0.7,
        presence_penalty: 0.1,
        frequency_penalty: 0.2,
        logprobs: true,
        response_format: %{type: "json"}
      ]

      {translated_opts, []} = Anthropic.translate_options(:chat, model, opts)

      # Should keep supported parameters
      assert Keyword.get(translated_opts, :temperature) == 0.7

      # Should remove unsupported parameters
      assert Keyword.get(translated_opts, :presence_penalty) == nil
      assert Keyword.get(translated_opts, :frequency_penalty) == nil
      assert Keyword.get(translated_opts, :logprobs) == nil
      assert Keyword.get(translated_opts, :response_format) == nil
    end
  end

  describe "usage extraction" do
    test "extract_usage with valid usage data" do
      model = ReqLLM.Model.from!("anthropic:claude-3-5-sonnet-20241022")

      body_with_usage = %{
        "usage" => %{
          "input_tokens" => 10,
          "output_tokens" => 20
        }
      }

      {:ok, usage} = Anthropic.extract_usage(body_with_usage, model)
      assert usage["input_tokens"] == 10
      assert usage["output_tokens"] == 20
    end

    test "extract_usage with missing usage data" do
      model = ReqLLM.Model.from!("anthropic:claude-3-5-sonnet-20241022")
      body_without_usage = %{"content" => []}

      {:error, :no_usage_found} = Anthropic.extract_usage(body_without_usage, model)
    end

    test "extract_usage with invalid body type" do
      model = ReqLLM.Model.from!("anthropic:claude-3-5-sonnet-20241022")

      {:error, :invalid_body} = Anthropic.extract_usage("invalid", model)
      {:error, :invalid_body} = Anthropic.extract_usage(nil, model)
      {:error, :invalid_body} = Anthropic.extract_usage(123, model)
    end
  end

  describe "stateful streaming with tool calls" do
    test "init_stream_state initializes empty tool call buffer" do
      model = ReqLLM.Model.from!("anthropic:claude-3-5-sonnet-20241022")
      state = Anthropic.init_stream_state(model)

      assert state == %{tool_calls: %{}}
    end

    test "decode_sse_event_stateful buffers tool call until complete" do
      model = ReqLLM.Model.from!("anthropic:claude-3-5-sonnet-20241022")
      initial_state = %{tool_calls: %{}}

      # Step 1: content_block_start with tool_use - should buffer, not emit
      tool_start_event = %{
        data: %{
          "type" => "content_block_start",
          "index" => 0,
          "content_block" => %{
            "type" => "tool_use",
            "id" => "toolu_123",
            "name" => "get_weather",
            "input" => %{}
          }
        }
      }

      {chunks1, state1} =
        ReqLLM.Providers.Anthropic.Response.decode_sse_event_stateful(
          tool_start_event,
          model,
          initial_state
        )

      # Should not emit any chunks yet
      assert chunks1 == []
      # Should buffer the tool call
      assert Map.has_key?(state1.tool_calls, 0)
      assert state1.tool_calls[0].name == "get_weather"
      assert state1.tool_calls[0].id == "toolu_123"
      assert state1.tool_calls[0].json_fragments == []

      # Step 2: input_json_delta events - should accumulate fragments
      json_delta1 = %{
        data: %{
          "type" => "content_block_delta",
          "index" => 0,
          "delta" => %{
            "type" => "input_json_delta",
            "partial_json" => "{\"location\":"
          }
        }
      }

      {chunks2, state2} =
        ReqLLM.Providers.Anthropic.Response.decode_sse_event_stateful(json_delta1, model, state1)

      assert chunks2 == []
      assert state2.tool_calls[0].json_fragments != []

      json_delta2 = %{
        data: %{
          "type" => "content_block_delta",
          "index" => 0,
          "delta" => %{
            "type" => "input_json_delta",
            "partial_json" => " \"San Francisco, CA\"}"
          }
        }
      }

      {chunks3, state3} =
        ReqLLM.Providers.Anthropic.Response.decode_sse_event_stateful(json_delta2, model, state2)

      assert chunks3 == []

      # Step 3: content_block_stop - should emit complete tool call
      stop_event = %{
        data: %{
          "type" => "content_block_stop",
          "index" => 0
        }
      }

      {chunks4, state4} =
        ReqLLM.Providers.Anthropic.Response.decode_sse_event_stateful(stop_event, model, state3)

      # Should emit one complete tool call chunk
      assert length(chunks4) == 1
      [tool_call_chunk] = chunks4
      assert tool_call_chunk.type == :tool_call
      assert tool_call_chunk.name == "get_weather"
      assert tool_call_chunk.arguments == %{"location" => "San Francisco, CA"}
      assert tool_call_chunk.metadata.id == "toolu_123"

      # Should remove from buffer
      assert state4.tool_calls == %{}
    end

    test "decode_sse_event_stateful emits text content immediately" do
      model = ReqLLM.Model.from!("anthropic:claude-3-5-sonnet-20241022")
      state = %{tool_calls: %{}}

      text_event = %{
        data: %{
          "type" => "content_block_delta",
          "index" => 0,
          "delta" => %{
            "type" => "text_delta",
            "text" => "Hello!"
          }
        }
      }

      {chunks, new_state} =
        ReqLLM.Providers.Anthropic.Response.decode_sse_event_stateful(text_event, model, state)

      # Text should be emitted immediately, not buffered
      assert length(chunks) == 1
      [text_chunk] = chunks
      assert text_chunk.type == :content
      assert text_chunk.text == "Hello!"
      # State should be unchanged
      assert new_state == state
    end

    test "flush_stream_state emits buffered incomplete tool calls" do
      model = ReqLLM.Model.from!("anthropic:claude-3-5-sonnet-20241022")

      # State with a partially completed tool call
      state = %{
        tool_calls: %{
          0 => %{
            id: "toolu_456",
            name: "incomplete_tool",
            index: 0,
            json_fragments: ["{\"partial\":"]
          }
        }
      }

      {chunks, _new_state} = Anthropic.flush_stream_state(model, state)

      # Should emit the tool call with empty arguments (invalid JSON)
      assert length(chunks) == 1
      [tool_call_chunk] = chunks
      assert tool_call_chunk.type == :tool_call
      assert tool_call_chunk.name == "incomplete_tool"
      assert tool_call_chunk.arguments == %{}
      assert tool_call_chunk.metadata.id == "toolu_456"
    end

    test "decode_sse_event_stateful normalizes usage metadata" do
      model = ReqLLM.Model.from!("anthropic:claude-3-5-sonnet-20241022")
      state = %{tool_calls: %{}}

      # message_delta with usage
      usage_event = %{
        data: %{
          "type" => "message_delta",
          "delta" => %{"stop_reason" => "tool_use"},
          "usage" => %{
            "input_tokens" => 100,
            "output_tokens" => 50
          }
        }
      }

      {chunks, _new_state} =
        ReqLLM.Providers.Anthropic.Response.decode_sse_event_stateful(usage_event, model, state)

      # Should emit meta chunks with usage and finish_reason
      assert length(chunks) == 2
      [usage_chunk, finish_chunk] = chunks

      assert usage_chunk.type == :meta
      assert Map.has_key?(usage_chunk.metadata, :usage)

      assert finish_chunk.type == :meta
      assert finish_chunk.metadata.finish_reason == :tool_calls
      assert finish_chunk.metadata.terminal? == true
    end
  end

  # Helper functions for Anthropic-specific fixtures

  defp anthropic_format_json_fixture(opts \\ []) do
    %{
      "id" => Keyword.get(opts, :id, "msg_01XFDUDYJgAACzvnptvVoYEL"),
      "type" => "message",
      "role" => "assistant",
      "model" => Keyword.get(opts, :model, "claude-3-5-sonnet-20241022"),
      "content" => [
        %{
          "type" => "text",
          "text" => Keyword.get(opts, :content, "Hello! I'm doing well, thank you for asking.")
        }
      ],
      "stop_reason" => Keyword.get(opts, :stop_reason, "stop"),
      "stop_sequence" => nil,
      "usage" => %{
        "input_tokens" => Keyword.get(opts, :input_tokens, 12),
        "output_tokens" => Keyword.get(opts, :output_tokens, 15)
      }
    }
  end
end
