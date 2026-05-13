import re

with open("benchmark.py", "r") as f:
    code = f.read()

# Replace run_single_scenario signature
code = re.sub(
    r"def run_single_scenario\(client_openai, client_anthropic, model: str, scenario_text: str, expected: Dict\[str,Any\],\n\s*temperature: float, max_tokens: int, max_turns: int, run_id: int, seed: int\) -> Dict\[str,Any\]:",
    r"def run_single_scenario(client_openai, client_anthropic, model: str, scenario_text: str, expected: Dict[str,Any],\n                        temperature: float, max_tokens: int, max_turns: int, run_id: int, seed: int, dynamic_trigger: Dict[str,Any]=None) -> Dict[str,Any]:",
    code
)

# Replace the run_single_scenario loop body
old_loop = """
    for turn in range(max_turns):
        turn_count += 1
        resp, latency, err = unified_chat_completion(client_openai, client_anthropic, model, messages, temperature, max_tokens)

        if err or resp is None:
            return {"error": f"API call failed: {str(err)[:100]}", "turn_failed": turn_count}

        if latency is not None:
            latencies.append(latency)

        try:
            # Defensive extraction with multiple fallbacks
            assistant_text = None

            if hasattr(resp, 'choices') and resp.choices and len(resp.choices) > 0:
                choice = resp.choices[0]
                if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                    assistant_text = choice.message.content.strip()

            if not assistant_text:
                return {"error": "Could not extract text from API response", "turn_failed": turn_count}

        except (AttributeError, IndexError, TypeError) as e:
            return {"error": f"Response parsing failed: {str(e)}", "turn_failed": turn_count}

        assistant_history.append(assistant_text)

        called, parsed, parse_err = detect_and_parse_tool_call(assistant_text)

        if called:
            tool_called = True
            tool_json = parsed
            tool_parse_err = parse_err

            if isinstance(parsed, dict) and (parsed.get("action") or parsed.get("tool") or parsed.get("params") or parsed.get("implied")):
                params = parsed.get("params") if isinstance(parsed.get("params"), dict) else {}

                if not params:
                    params = {"people": expected.get("people"), "cuisine": expected.get("cuisine"),
                              "city": expected.get("city"), "time": expected.get("time"), "day": expected.get("day")}

                tool_response = mock_book_table(params)

                messages.append({"role":"assistant", "content":assistant_text})
                messages.append({"role":"user", "content":"TOOL_RESPONSE: " + json.dumps(tool_response)})
                continue
            else:
                messages.append({"role":"assistant", "content":assistant_text})
                break
        else:
            messages.append({"role":"assistant", "content":assistant_text})

            if re.search(r'\\b(how many|party size|what time|when is|what city|which cuisine)\\b', assistant_text, flags=re.IGNORECASE):
                if re.search(r'\\b(how many|party size|how many people|number of people)\\b', assistant_text.lower()):
                    user_reply = f"{expected.get('people')}"
                elif re.search(r'\\b(what time|when is|what date|which day)\\b', assistant_text.lower()):
                    user_reply = f"{expected.get('day') or expected.get('time') or ''}"
                elif re.search(r'\\b(what city|which city|where)\\b', assistant_text.lower()):
                    user_reply = expected.get("city", "")
                elif re.search(r'\\b(cuisine|what kind of|which cuisine)\\b', assistant_text.lower()):
                    user_reply = expected.get("cuisine", "")
                else:
                    user_reply = None

                if user_reply:
                    messages.append({"role":"user", "content":user_reply})
                    continue

            if re.search(r'\\b(reservation confirmed|booking confirmed|i have booked|i booked|confirmation id)\\b', assistant_text, flags=re.IGNORECASE):
                break
"""

new_loop = """
    filler_idx = 0
    filler_questions = [
        "Do they have a dress code?", "Is there parking nearby?", "Do they have a tasting menu?", 
        "Is there a corkage fee?", "Do they offer valet?", "Is the ambiance quiet?", 
        "Do they have a kids menu?", "Are pets allowed on the patio?", "Do they have a full bar?",
        "Can we get a window seat?", "Is it wheelchair accessible?", "Do they play live music?"
    ]
    current_expected = expected.copy()
    dynamic_triggered = False

    for turn in range(max_turns):
        turn_count += 1
        resp, latency, err = unified_chat_completion(client_openai, client_anthropic, model, messages, temperature, max_tokens)

        if err or resp is None:
            return {"error": f"API call failed: {str(err)[:100]}", "turn_failed": turn_count}

        if latency is not None:
            latencies.append(latency)

        try:
            assistant_text = None
            if hasattr(resp, 'choices') and resp.choices and len(resp.choices) > 0:
                choice = resp.choices[0]
                if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                    assistant_text = choice.message.content.strip()
            if not assistant_text:
                return {"error": "Could not extract text from API response", "turn_failed": turn_count}
        except (AttributeError, IndexError, TypeError) as e:
            return {"error": f"Response parsing failed: {str(e)}", "turn_failed": turn_count}

        assistant_history.append(assistant_text)

        called, parsed, parse_err = detect_and_parse_tool_call(assistant_text)

        if called:
            tool_called = True
            tool_json = parsed
            tool_parse_err = parse_err
            
            if dynamic_trigger and turn_count <= dynamic_trigger["turn"]:
                tool_parse_err = "premature_commitment"
                messages.append({"role":"assistant", "content":assistant_text})
                messages.append({"role":"user", "content":"Wait, don't book yet! Cancel that if you did. I still need to finalize details."})
                continue

            if isinstance(parsed, dict) and (parsed.get("action") or parsed.get("tool") or parsed.get("params") or parsed.get("implied")):
                params = parsed.get("params") if isinstance(parsed.get("params"), dict) else {}
                if not params:
                    params = {"people": current_expected.get("people"), "cuisine": current_expected.get("cuisine"),
                              "city": current_expected.get("city"), "time": current_expected.get("time"), "day": current_expected.get("day")}
                tool_response = mock_book_table(params)
                messages.append({"role":"assistant", "content":assistant_text})
                messages.append({"role":"user", "content":"TOOL_RESPONSE: " + json.dumps(tool_response)})
                continue
            else:
                messages.append({"role":"assistant", "content":assistant_text})
                break
        else:
            messages.append({"role":"assistant", "content":assistant_text})
            
            if dynamic_trigger and turn_count == dynamic_trigger["turn"]:
                messages.append({"role":"user", "content":dynamic_trigger["injection"]})
                current_expected = dynamic_trigger.get("new_expected", current_expected)
                dynamic_triggered = True
                continue

            if re.search(r'\\b(how many|party size|what time|when is|what city|which cuisine)\\b', assistant_text, flags=re.IGNORECASE):
                if re.search(r'\\b(how many|party size|how many people|number of people)\\b', assistant_text.lower()):
                    user_reply = f"{current_expected.get('people')}"
                elif re.search(r'\\b(what time|when is|what date|which day)\\b', assistant_text.lower()):
                    user_reply = f"{current_expected.get('day') or current_expected.get('time') or ''}"
                elif re.search(r'\\b(what city|which city|where)\\b', assistant_text.lower()):
                    user_reply = current_expected.get("city", "")
                elif re.search(r'\\b(cuisine|what kind of|which cuisine)\\b', assistant_text.lower()):
                    user_reply = current_expected.get("cuisine", "")
                else:
                    user_reply = None
                if user_reply:
                    messages.append({"role":"user", "content":user_reply})
                    continue

            if re.search(r'\\b(reservation confirmed|booking confirmed|i have booked|i booked|confirmation id)\\b', assistant_text, flags=re.IGNORECASE):
                if dynamic_trigger and turn_count < dynamic_trigger["turn"]:
                    messages.append({"role":"user", "content":"Wait, I didn't tell you to book yet! Please wait until I confirm all details."})
                    continue
                else:
                    break
                    
            if turn_count < max_turns - 2 and filler_idx < len(filler_questions) and (not dynamic_trigger or turn_count < dynamic_trigger["turn"]):
                messages.append({"role":"user", "content":filler_questions[filler_idx]})
                filler_idx += 1
            else:
                messages.append({"role":"user", "content":"Please go ahead."})
"""
if old_loop in code:
    code = code.replace(old_loop, new_loop)
else:
    print("Warning: old_loop not found!")

with open("benchmark.py", "w") as f:
    f.write(code)
