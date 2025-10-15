defmodule ReqLLM.Providers.Anthropic.Response do
  @moduledoc """
  Anthropic-specific response decoding for the Messages API format.

  Handles decoding Anthropic Messages API responses to ReqLLM structures.

  ## Anthropic Response Format

      %{
        "id" => "msg_01XFDUDYJgAACzvnptvVoYEL",
        "type" => "message",
        "role" => "assistant",
        "model" => "claude-3-5-sonnet-20241022",
        "content" => [
          %{"type" => "text", "text" => "Hello! How can I help you today?"}
        ],
        "stop_reason" => "stop",
        "stop_sequence" => nil,
        "usage" => %{
          "input_tokens" => 10,
          "output_tokens" => 20
        }
      }

  ## Streaming Format

  Anthropic uses Server-Sent Events (SSE) with different event types:
  - message_start: Initial message metadata
  - content_block_start: Start of content block
  - content_block_delta: Incremental content
  - content_block_stop: End of content block
  - message_delta: Final message updates
  - message_stop: End of message

  """

  @doc """
  Decode Anthropic response data to ReqLLM.Response.
  """
  @spec decode_response(map(), ReqLLM.Model.t()) :: {:ok, ReqLLM.Response.t()} | {:error, term()}
  def decode_response(data, model) when is_map(data) do
    id = Map.get(data, "id", "unknown")
    model_name = Map.get(data, "model", model.model || "unknown")
    usage = parse_usage(Map.get(data, "usage"))

    finish_reason = parse_finish_reason(Map.get(data, "stop_reason"))

    content_chunks = decode_content(Map.get(data, "content", []))
    message = build_message_from_chunks(content_chunks)

    context = %ReqLLM.Context{
      messages: if(message, do: [message], else: [])
    }

    response = %ReqLLM.Response{
      id: id,
      model: model_name,
      context: context,
      message: message,
      stream?: false,
      stream: nil,
      usage: usage,
      finish_reason: finish_reason,
      provider_meta: Map.drop(data, ["id", "model", "content", "usage", "stop_reason"])
    }

    {:ok, response}
  end

  def decode_response(_data, _model) do
    {:error, :not_implemented}
  end

  @doc """
  Decode Anthropic SSE event data into StreamChunks (stateless version).

  This is kept for backward compatibility but the stateful version is now preferred.
  """
  @spec decode_sse_event(map(), ReqLLM.Model.t()) :: [ReqLLM.StreamChunk.t()]
  def decode_sse_event(%{data: data}, _model) when is_map(data) do
    case data do
      %{"type" => "content_block_delta", "index" => index, "delta" => delta} ->
        decode_content_block_delta(delta, index)

      %{"type" => "content_block_start", "index" => index, "content_block" => block} ->
        decode_content_block_start(block, index)

      # Terminal events with metadata
      %{"type" => "message_stop"} ->
        [ReqLLM.StreamChunk.meta(%{terminal?: true})]

      %{"type" => "message_delta", "delta" => delta} ->
        finish_reason =
          case Map.get(delta, "stop_reason") do
            "end_turn" -> :stop
            "max_tokens" -> :length
            "stop_sequence" -> :stop
            "tool_use" -> :tool_calls
            _ -> :unknown
          end

        usage = Map.get(data, "usage", %{})

        chunks = [ReqLLM.StreamChunk.meta(%{finish_reason: finish_reason, terminal?: true})]

        # Add usage chunk if present
        if usage == %{} do
          chunks
        else
          usage_chunk = ReqLLM.StreamChunk.meta(%{usage: usage})
          [usage_chunk | chunks]
        end

      %{"type" => "ping"} ->
        # Keep-alive ping, no content
        []

      _ ->
        []
    end
  end

  def decode_sse_event(_, _model), do: []

  @doc """
  Decode Anthropic SSE event data into StreamChunks with stateful tool call buffering.

  This version buffers tool calls until all JSON fragments are received, then emits
  a complete :tool_call chunk with parsed arguments. This provides compatibility
  with providers like Google that send complete tool calls in one chunk.
  """
  @spec decode_sse_event_stateful(map(), ReqLLM.Model.t(), map()) ::
          {[ReqLLM.StreamChunk.t()], map()}
  def decode_sse_event_stateful(%{data: data}, _model, provider_state) when is_map(data) do
    case data do
      %{"type" => "content_block_delta", "index" => index, "delta" => delta} ->
        handle_content_block_delta_stateful(delta, index, provider_state)

      %{"type" => "content_block_start", "index" => index, "content_block" => block} ->
        handle_content_block_start_stateful(block, index, provider_state)

      %{"type" => "content_block_stop", "index" => index} ->
        handle_content_block_stop_stateful(index, provider_state)

      # Terminal events with metadata
      %{"type" => "message_stop"} ->
        {[ReqLLM.StreamChunk.meta(%{terminal?: true})], provider_state}

      %{"type" => "message_delta", "delta" => delta} ->
        finish_reason =
          case Map.get(delta, "stop_reason") do
            "end_turn" -> :stop
            "max_tokens" -> :length
            "stop_sequence" -> :stop
            "tool_use" -> :tool_calls
            _ -> :unknown
          end

        usage = Map.get(data, "usage", %{})

        chunks = [ReqLLM.StreamChunk.meta(%{finish_reason: finish_reason, terminal?: true})]

        # Add usage chunk if present
        chunks =
          if usage == %{} do
            chunks
          else
            usage_chunk = ReqLLM.StreamChunk.meta(%{usage: usage})
            [usage_chunk | chunks]
          end

        {chunks, provider_state}

      %{"type" => "ping"} ->
        # Keep-alive ping, no content
        {[], provider_state}

      _ ->
        {[], provider_state}
    end
  end

  def decode_sse_event_stateful(_, _model, provider_state), do: {[], provider_state}

  # Stateful handlers

  defp handle_content_block_start_stateful(%{"type" => "text"} = block, index, provider_state) do
    # Text content - emit immediately
    chunks = decode_content_block_start(block, index)
    {chunks, provider_state}
  end

  defp handle_content_block_start_stateful(%{"type" => "thinking"} = block, index, provider_state) do
    # Thinking content - emit immediately
    chunks = decode_content_block_start(block, index)
    {chunks, provider_state}
  end

  defp handle_content_block_start_stateful(
         %{"type" => "tool_use", "id" => id, "name" => name},
         index,
         provider_state
       ) do
    # Tool call start - buffer it, don't emit yet
    tool_call = %{
      id: id,
      name: name,
      index: index,
      json_fragments: []
    }

    new_tool_calls = Map.put(provider_state.tool_calls, index, tool_call)
    new_state = %{provider_state | tool_calls: new_tool_calls}

    {[], new_state}
  end

  defp handle_content_block_start_stateful(_block, _index, provider_state) do
    {[], provider_state}
  end

  defp handle_content_block_delta_stateful(
         %{"type" => "text_delta", "text" => text},
         _index,
         provider_state
       )
       when is_binary(text) do
    # Text delta - emit immediately
    {[ReqLLM.StreamChunk.text(text)], provider_state}
  end

  defp handle_content_block_delta_stateful(
         %{"type" => "thinking_delta", "thinking" => text},
         _index,
         provider_state
       )
       when is_binary(text) do
    {[ReqLLM.StreamChunk.thinking(text)], provider_state}
  end

  defp handle_content_block_delta_stateful(
         %{"type" => "thinking_delta", "text" => text},
         _index,
         provider_state
       )
       when is_binary(text) do
    {[ReqLLM.StreamChunk.thinking(text)], provider_state}
  end

  defp handle_content_block_delta_stateful(
         %{"type" => "input_json_delta", "partial_json" => fragment},
         index,
         provider_state
       )
       when is_binary(fragment) do
    # JSON fragment for tool call - accumulate in buffer
    case Map.get(provider_state.tool_calls, index) do
      nil ->
        # No tool call started for this index, ignore
        {[], provider_state}

      tool_call ->
        updated_tool_call = %{tool_call | json_fragments: [tool_call.json_fragments, fragment]}
        new_tool_calls = Map.put(provider_state.tool_calls, index, updated_tool_call)
        new_state = %{provider_state | tool_calls: new_tool_calls}
        {[], new_state}
    end
  end

  defp handle_content_block_delta_stateful(_, _index, provider_state), do: {[], provider_state}

  defp handle_content_block_stop_stateful(index, provider_state) do
    # Content block stopped - if it's a tool call, emit it now
    case Map.get(provider_state.tool_calls, index) do
      nil ->
        # Not a tool call, nothing to emit
        {[], provider_state}

      tool_call ->
        # Parse the accumulated JSON and emit complete tool call
        arguments =
          if tool_call.json_fragments == [] do
            %{}
          else
            json_str = IO.iodata_to_binary(tool_call.json_fragments)

            case Jason.decode(json_str) do
              {:ok, args} -> args
              {:error, _} -> %{}
            end
          end

        chunk = ReqLLM.StreamChunk.tool_call(tool_call.name, arguments, %{id: tool_call.id})

        # Remove from buffer
        new_tool_calls = Map.delete(provider_state.tool_calls, index)
        new_state = %{provider_state | tool_calls: new_tool_calls}

        {[chunk], new_state}
    end
  end

  # Private helper functions

  defp decode_content([]), do: []

  defp decode_content(content) when is_list(content) do
    content
    |> Enum.map(&decode_content_block/1)
    |> List.flatten()
    |> Enum.reject(&is_nil/1)
  end

  defp decode_content(content) when is_binary(content) do
    [ReqLLM.StreamChunk.text(content)]
  end

  defp decode_content_block(%{"type" => "text", "text" => text}) do
    ReqLLM.StreamChunk.text(text)
  end

  defp decode_content_block(%{"type" => "thinking", "thinking" => text}) do
    ReqLLM.StreamChunk.thinking(text)
  end

  defp decode_content_block(%{"type" => "thinking", "text" => text}) do
    ReqLLM.StreamChunk.thinking(text)
  end

  defp decode_content_block(%{"type" => "tool_use", "id" => id, "name" => name, "input" => input}) do
    ReqLLM.StreamChunk.tool_call(name, input, %{id: id})
  end

  defp decode_content_block(_), do: nil

  defp decode_content_block_delta(%{"type" => "text_delta", "text" => text}, _index)
       when is_binary(text) do
    [ReqLLM.StreamChunk.text(text)]
  end

  defp decode_content_block_delta(%{"type" => "thinking_delta", "thinking" => text}, _index)
       when is_binary(text) do
    [ReqLLM.StreamChunk.thinking(text)]
  end

  defp decode_content_block_delta(%{"type" => "thinking_delta", "text" => text}, _index)
       when is_binary(text) do
    [ReqLLM.StreamChunk.thinking(text)]
  end

  defp decode_content_block_delta(
         %{"type" => "input_json_delta", "partial_json" => fragment},
         index
       )
       when is_binary(fragment) do
    # Accumulate JSON fragments; StreamResponse.extract_tool_calls will merge these
    [ReqLLM.StreamChunk.meta(%{tool_call_args: %{index: index, fragment: fragment}})]
  end

  defp decode_content_block_delta(_, _index), do: []

  defp decode_content_block_start(%{"type" => "text", "text" => text}, _index) do
    [ReqLLM.StreamChunk.text(text)]
  end

  defp decode_content_block_start(%{"type" => "thinking", "thinking" => text}, _index) do
    [ReqLLM.StreamChunk.thinking(text)]
  end

  defp decode_content_block_start(%{"type" => "thinking", "text" => text}, _index) do
    [ReqLLM.StreamChunk.thinking(text)]
  end

  defp decode_content_block_start(%{"type" => "tool_use", "id" => id, "name" => name}, index) do
    # Tool call start - send empty arguments that will be filled by deltas
    [ReqLLM.StreamChunk.tool_call(name, %{}, %{id: id, index: index, start: true})]
  end

  defp decode_content_block_start(_, _index), do: []

  defp build_message_from_chunks([]), do: nil

  defp build_message_from_chunks(chunks) do
    content_parts =
      chunks
      |> Enum.filter(&(&1.type in [:content, :thinking]))
      |> Enum.map(&chunk_to_content_part/1)
      |> Enum.reject(&is_nil/1)

    tool_calls =
      chunks
      |> Enum.filter(&(&1.type == :tool_call))
      |> Enum.map(&chunk_to_tool_call/1)
      |> Enum.reject(&is_nil/1)

    if content_parts != [] or tool_calls != [] do
      %ReqLLM.Message{
        role: :assistant,
        content: content_parts,
        tool_calls: if(tool_calls != [], do: tool_calls),
        metadata: %{}
      }
    end
  end

  defp chunk_to_content_part(%ReqLLM.StreamChunk{type: :content, text: text}) do
    %ReqLLM.Message.ContentPart{type: :text, text: text}
  end

  defp chunk_to_content_part(%ReqLLM.StreamChunk{type: :thinking, text: text}) do
    %ReqLLM.Message.ContentPart{type: :thinking, text: text}
  end

  defp chunk_to_content_part(_), do: nil

  defp chunk_to_tool_call(%ReqLLM.StreamChunk{
         type: :tool_call,
         name: name,
         arguments: args,
         metadata: meta
       }) do
    args_json = if is_binary(args), do: args, else: Jason.encode!(args)
    id = Map.get(meta, :id)
    ReqLLM.ToolCall.new(id, name, args_json)
  end

  defp chunk_to_tool_call(_), do: nil

  defp parse_usage(%{"input_tokens" => input, "output_tokens" => output} = usage) do
    cached_tokens = Map.get(usage, "cache_read_input_tokens", 0)
    reasoning_tokens = Map.get(usage, "reasoning_output_tokens", 0)

    %{
      input_tokens: input,
      output_tokens: output,
      total_tokens: input + output,
      cached_tokens: cached_tokens,
      reasoning_tokens: reasoning_tokens
    }
  end

  defp parse_usage(_),
    do: %{
      input_tokens: 0,
      output_tokens: 0,
      total_tokens: 0,
      cached_tokens: 0,
      reasoning_tokens: 0
    }

  defp parse_finish_reason("stop"), do: :stop
  defp parse_finish_reason("max_tokens"), do: :length
  defp parse_finish_reason("tool_use"), do: :tool_calls
  defp parse_finish_reason("end_turn"), do: :stop
  defp parse_finish_reason("content_filter"), do: :content_filter
  defp parse_finish_reason(reason) when is_binary(reason), do: :error
  defp parse_finish_reason(_), do: nil
end
