"""Run the native event adapter through Hermes' stream consumer and watchdog."""
import pytest

import test_directsdk as native_fixture


def test_hermes_stream_matches_nonstream_and_replays(tmp_path, monkeypatch, profile):
    from agent.turn_api_call import _should_stream
    from agent.usage_pricing import normalize_usage
    from run_agent import AIAgent

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    client = native_fixture.Contract().client(str(tmp_path))
    agent = AIAgent(api_key="test-key", provider="openrouter", model="test/model",
                    base_url="https://openrouter.ai/api/v1", quiet_mode=True,
                    skip_context_files=True, skip_memory=True)
    agent.provider, agent.base_url = profile.name, client.base_url
    monkeypatch.setattr("providers.get_provider_profile", lambda name: profile)
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **kwargs: client)
    monkeypatch.setattr(agent, "_close_request_openai_client", lambda *args, **kwargs: None)
    assert _should_stream(agent)
    request = native_fixture.Contract().request()
    try:
        plain = client.create(**request)
        streamed = agent._interruptible_streaming_api_call(request)
        def message(response):
            msg = response.choices[0].message
            return {"role": msg.role, "content": msg.content,
                    "reasoning_content": msg.reasoning_content,
                    "reasoning_details": msg.reasoning_details,
                    "tool_calls": [{"id": tc.id, "type": tc.type,
                                    "function": {"name": tc.function.name,
                                                 "arguments": tc.function.arguments}}
                                   for tc in msg.tool_calls or []]}
        assert message(streamed) == message(plain)
        assert streamed.choices[0].finish_reason == plain.choices[0].finish_reason
        assert normalize_usage(streamed.usage, api_mode="chat_completions") == normalize_usage(plain.usage, api_mode="chat_completions")
        # Native signed history and the exact tool arguments survive the real accumulator.
        request["messages"] += [message(streamed), {"role": "tool", "tool_call_id": "toolu_test", "content": "host result"}]
        assert agent._interruptible_streaming_api_call(request).choices[0].message.content == "done"
        assert not getattr(agent, "_disable_streaming", False)
    finally:
        client.close()


@pytest.mark.parametrize("interrupt", [False, True])
def test_native_tool_argument_progress_then_silence_or_interrupt(tmp_path, monkeypatch, interrupt):
    from agent import chat_completion_helpers as h
    from run_agent import AIAgent

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    client = native_fixture.Contract().client(str(tmp_path))
    request = native_fixture.Contract().request()
    script = tmp_path / "native.py"
    # Emit only native argument events; no text/thinking can accidentally keep this alive.
    script.write_text("""import json, pathlib, subprocess, sys, time
if '--version' in sys.argv:
    print('2.1.263 (Claude Code)'); sys.exit()
for line in sys.stdin:
    pass
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
pathlib.Path(__file__).with_suffix('.child').write_text(str(child.pid))
for i in range(8):
    print(json.dumps({'type':'stream_event','event':{'type':'content_block_delta','index':0,'delta':{'type':'input_json_delta','partial_json':'x' * 4096}}}), flush=True)
print(json.dumps({'type':'system','subtype':'status','status':None}), flush=True)
time.sleep(60)
""")
    # Deterministic clock: native events each represent 60s of generation. The real
    # monitor must tolerate >240s total, then kill after >240s with no event.
    agent = AIAgent(api_key="test-key", provider="openrouter", model="test/model",
                    base_url="https://openrouter.ai/api/v1", quiet_mode=True,
                    skip_context_files=True, skip_memory=True, enabled_toolsets=[])
    agent.base_url = client.base_url
    now = [1000.0]
    monkeypatch.setattr(h.time, "time", lambda: now[0])
    call = h._StreamingCall(agent, request, None)
    call.clients.set_client(client)
    call._monitor_interrupted = {"yes": False}
    attempt = call._start_stream_attempt()
    call._stream_stale_timeout = 240.0
    kills = []
    stream = client.create(**request, stream=True, timeout=2)
    native_pid = None
    class Done:
        steps = 0
        def is_set(self):
            return bool(kills)
        def wait(self, timeout):
            nonlocal native_pid
            self.steps += 1
            now[0] += 60
            if self.steps <= 9:
                chunk = next(stream)
                native_pid = stream.request.process.pid
                assert chunk.choices[0].delta.tool_calls is None
                assert call._accept_chat_chunk(attempt, chunk)
            elif interrupt:
                agent._interrupt_requested = True
    call._call_done = Done()
    method = "_abort_for_interrupt" if interrupt else "_kill_stale_stream"
    abort = getattr(call, method)
    def stop(elapsed):
        abort(elapsed)  # real Hermes registry -> client.cancel -> process-group kill
        kills.append(elapsed)
    monkeypatch.setattr(call, method, stop)
    try:
        call._monitor_loop()
        assert kills == ([60.0] if interrupt else [300.0])
        assert now[0] > 1000 + 240  # sustained generation outlives the old watchdog
        assert stream.request.cancelled.is_set()
        assert call._stream_attempt_was_cancelled(attempt)
        assert native_fixture._wait_gone(native_pid, int(script.with_suffix('.child').read_text()))
    finally:
        stream.close()
        client.close()
