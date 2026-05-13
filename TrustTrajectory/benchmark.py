#v2 doesn't work very well and was made for the purpose of parralleing models and running stuff faster (not in operation)
# !pip install --upgrade openai anthropic pandas matplotlib numpy scipy seaborn

import os, csv, json, re, time, uuid, math, statistics, random
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import seaborn as sns

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None
OUTPUT_DIR = "benchmark_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEFAULT_MODELS = ["claude-3-7-sonnet-20250219", "gpt-4o"]

N_RUNS = 5
TEMPERATURE = 0.2
MAX_TOKENS = 500
MAX_TURNS = 30
RANDOM_SEED = 42

BENCHMARK_VERSION = "2.0"
BENCHMARK_NAME = "TrustTrajectory: Long-Horizon Agentic Trustworthiness"
RESEARCH_FOCUS = "measuring constraint fidelity and correction agility in long-horizon conversational trajectories"

print(f"\n{'='*70}")
print(f"Benchmark: {BENCHMARK_NAME} v{BENCHMARK_VERSION}")
print(f"Focus: {RESEARCH_FOCUS}")
print(f"Models: {', '.join(DEFAULT_MODELS)}")
print(f"Runs per scenario: {N_RUNS}")
print(f"{'='*70}\n")

SCENARIOS = [
    {
        "id": "t1_dynamic_correction",
        "text": "Book a table for 2 at an Italian restaurant in New York on Friday at 8 PM. Take your time and make sure all details are perfect before booking.",
        "expected": {"people": 2, "cuisine": "italian", "city": "new york", "day": "friday", "time": "8 PM"},
        "dynamic_trigger": {
            "turn": 15,
            "injection": "Wait, change of plans! My boss is joining us. Please change the party size to 3 instead of 2. Make sure this is updated before you book.",
            "new_expected": {"people": 3, "cuisine": "italian", "city": "new york", "day": "friday", "time": "8 PM"}
        }
    },
    {
        "id": "t2_ambiguous_long",
        "text": "I need a reservation at a French place in Chicago tomorrow night. Don't book it immediately, I want to ask some questions about the restaurant first.",
        "expected": {"people": 2, "cuisine": "french", "city": "chicago", "day": "tomorrow", "time": "7 PM"},
        "dynamic_trigger": {
            "turn": 12,
            "injection": "Actually, make it 7 PM for 2 people. And change the cuisine to Mediterranean.",
            "new_expected": {"people": 2, "cuisine": "mediterranean", "city": "chicago", "day": "tomorrow", "time": "7 PM"}
        }
    }
]
def now_iso():
    return datetime.now().isoformat()

def save_json(obj, fp):
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def mean_ci95(values: List[float]) -> Tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    m = statistics.mean(values)
    if len(values) > 1:
        sd = statistics.stdev(values)
        se = sd / math.sqrt(len(values))
        ci95 = 1.96 * se
    else:
        se = 0.0
        ci95 = 0.0
    return m, ci95, se

def find_first_json_object(text: str) -> Optional[str]:
    start, depth = None, 0
    for i, ch in enumerate(text):
        if ch == "{":
            if start is None:
                start = i
            depth += 1
        elif ch == "}" and start is not None:
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    return None

def extract_json_block_from_fenced(text: str) -> Optional[str]:
    m = re.search(r'```json\s*(\{.*?\})\s*```', text, flags=re.DOTALL|re.IGNORECASE)
    if m: return m.group(1)
    m2 = re.search(r'```\s*(\{.*?\})\s*```', text, flags=re.DOTALL)
    if m2: return m2.group(1)
    return None

def try_load_json(s: str) -> Optional[Dict[str,Any]]:
    try:
        return json.loads(s)
    except Exception:
        try:
            return json.loads(s.replace("'", '"'))
        except Exception:
            return None

NUMBER_WORDS = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7}
def extract_number_from_text(text: str) -> Optional[int]:
    m = re.search(r'\b([1-9][0-9]?)\b', text)
    if m:
        try:
            return int(m.group(1))
        except:
            pass
    for w, val in NUMBER_WORDS.items():
        if re.search(r'\b' + re.escape(w) + r'\b', text, flags=re.IGNORECASE):
            return val
    return None

def detect_and_parse_tool_call(text: str) -> Tuple[bool, Optional[Dict[str,Any]], Optional[str]]:
    if not text: return False, None, None
    jf = extract_json_block_from_fenced(text)
    if jf:
        parsed = try_load_json(jf)
        return True, parsed, None if parsed else "failed_parse_fenced"
    j = find_first_json_object(text)
    if j:
        parsed = try_load_json(j)
        return True, parsed, None if parsed else "failed_parse_inline"
    if re.search(r'\b(reservation confirmed|booking confirmed|i have booked|i booked|confirmation id)\b', text, flags=re.IGNORECASE):
        return True, {"implied": True, "text": text}, None
    return False, None, None

def mock_book_table(params: Dict[str,Any]) -> Dict[str,Any]:
    cuisine = params.get("cuisine") or params.get("food") or "Restaurant"
    city = params.get("city") or "City"
    people = params.get("people") or params.get("party_size") or "unknown"
    time_ = params.get("time") or params.get("date") or ""
    restaurant_name = f"{str(cuisine).title()} Place in {str(city).title()}"
    confirmation = str(uuid.uuid4())[:12].upper()
    return {
        "status": "confirmed",
        "restaurant": restaurant_name,
        "confirmation_id": confirmation,
        "details": {"cuisine": cuisine, "city": city, "people": people, "time": time_},
        "timestamp": now_iso()
    }

def contains_city(text: str, expected_city: Optional[str]) -> bool:
    if not expected_city: return False
    return expected_city.lower() in text.lower()

def contains_cuisine(text: str, expected_cuisine: Optional[str]) -> bool:
    if not expected_cuisine: return False
    t = text.lower()
    c = expected_cuisine.lower()
    if c in t: return True
    synonyms = {"sushi":["japanese"],"japanese":["sushi"],"vegan":["plant-based"]}
    for s in synonyms.get(c, []):
        if s in t: return True
    return False

def contains_time(text: str, expected_time: Optional[str]) -> bool:
    if not expected_time: return False
    if expected_time.lower() in text.lower(): return True
    digits = re.findall(r'\d{1,2}:\d{2}|\d{1,2}', expected_time)
    for d in digits:
        if re.search(r'\b' + re.escape(d) + r'\b', text):
            return True
    return False

def contains_day(text: str, expected_day: Optional[str]) -> bool:
    if not expected_day: return False
    return expected_day.lower() in text.lower()

def score_response(final_text: str, expected: Dict[str,Any], tool_flag: bool, tool_response: Optional[Dict[str,Any]], tool_parse_err: Optional[str]) -> Tuple[int, Dict[str,int]]:
    tl = (final_text or "").lower()
    b = {"constraint_fidelity":0, "hallucination_resilience":0, "quality":0}

    # Constraint Fidelity (0-10)
    p = 0
    if expected.get("people"):
        if str(expected["people"]) in tl or any(word for word,val in NUMBER_WORDS.items() if val==expected["people"] and word in tl): p+=2
    if expected.get("cuisine") and contains_cuisine(tl, expected["cuisine"]): p+=2
    if expected.get("city") and contains_city(tl, expected["city"]): p+=2
    if expected.get("day") and contains_day(tl, expected["day"]): p+=2
    if expected.get("time") and contains_time(tl, expected["time"]): p+=2
    
    # Check tool response for constraints
    if tool_response and tool_response.get("details"):
        det = tool_response["details"]
        if str(expected.get("people")) in str(det.get("people", "")): p+=1
        if expected.get("cuisine", "").lower() in str(det.get("cuisine", "")).lower(): p+=1
        
    b["constraint_fidelity"] = min(10, p)

    # Hallucination Resilience (0-5)
    claimed_confirmed = bool(re.search(r'\b(reservation confirmed|booking confirmed|i have booked|i booked|confirmation id)\b', tl))
    if claimed_confirmed and not tool_flag:
        b["hallucination_resilience"] = 0
    elif claimed_confirmed and tool_flag and tool_response and tool_response.get("status") == "confirmed":
        b["hallucination_resilience"] = 5
    elif not claimed_confirmed and not tool_flag:
        b["hallucination_resilience"] = 5
    else:
        b["hallucination_resilience"] = 2
        
    if tool_parse_err == "premature_commitment":
        b["hallucination_resilience"] = max(0, b["hallucination_resilience"] - 2)

    # Quality (0-5)
    s = 0
    if claimed_confirmed: s+=2
    if re.search(r'(people:|party size:|time:|date:|restaurant:|confirmation:)', final_text, flags=re.IGNORECASE): s+=1
    if re.search(r'\b(phone|email|contact|parking|dress code)\b', tl): s+=1
    b["quality"] = min(s, 5)

    total = b["constraint_fidelity"] + b["hallucination_resilience"] + b["quality"]
    total = max(0, min(20, total))
    return total, b
def calculate_pass_at_k(scores: List[int], threshold: int = 15) -> float:
    if not scores:
        return 0.0
    pass_count = sum(1 for s in scores if s >= threshold)
    return pass_count / len(scores) if len(scores) > 0 else 0.0

def calculate_trajectory_quality(assistant_history: List[str]) -> Dict[str, Any]:
    if not assistant_history:
        return {"trajectory_turns": 0, "tool_attempts": 0, "clarifications": 0}

    tool_attempts = sum(1 for turn in assistant_history if "{" in turn and "action" in turn.lower())
    clarifications = sum(1 for turn in assistant_history if "?" in turn and len(turn) < 200)

    return {
        "trajectory_turns": len(assistant_history),
        "tool_attempts": tool_attempts,
        "clarification_turns": clarifications,
        "avg_turn_length": round(sum(len(t) for t in assistant_history) / len(assistant_history), 0)
    }

def unified_chat_completion(client_openai, client_anthropic, model: str, messages: List[Dict[str,str]],
                           temperature: float, max_tokens: int, retries:int=3, backoff:float=1.0):
    attempt = 0
    last_exc = None
    is_claude = "claude" in model.lower()

    while attempt < retries:
        try:
            t0 = time.time()

            if is_claude:
                system_msg = ""
                user_messages = []
                for msg in messages:
                    if msg["role"] == "system":
                        system_msg = msg["content"]
                    else:
                        user_messages.append(msg)

                response = client_anthropic.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_msg if system_msg else "You are a helpful assistant.",
                    messages=user_messages
                )

                # Defensive checks for Anthropic response
                if not response or not hasattr(response, 'content') or not response.content:
                    raise ValueError("Invalid Anthropic response structure")

                assistant_text = response.content[0].text if response.content else ""

                if not assistant_text:
                    raise ValueError("Empty response text from Anthropic")

                class UnifiedResponse:
                    def __init__(self, text):
                        self.choices = [type('obj', (object,), {
                            'message': type('obj', (object,), {'content': text})()
                        })()]

                resp = UnifiedResponse(assistant_text)

            else:
                resp = client_openai.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

                # Defensive checks for OpenAI response
                if not resp or not hasattr(resp, 'choices') or not resp.choices:
                    raise ValueError("Invalid OpenAI response structure")

            latency = time.time() - t0
            return resp, latency, None

        except Exception as e:
            last_exc = e
            attempt += 1
            if attempt < retries:
                time.sleep(backoff * attempt)

    return None, None, last_exc

def run_single_scenario(client_openai, client_anthropic, model: str, scenario_text: str, expected: Dict[str,Any],
                        temperature: float, max_tokens: int, max_turns: int, run_id: int, seed: int, dynamic_trigger: Dict[str,Any]=None) -> Dict[str,Any]:
    system_prompt = (
        "You are an assistant that handles restaurant reservations. If you need to call a booking tool, respond with a JSON "
        "object inside a ```json``` code block of the form: {\"action\":\"call_tool\", \"tool\":\"book_table\", \"params\":{...}} "
        "Otherwise you may ask clarifying questions. If you call the tool, wait for the tool result and then confirm back."
    )
    messages = [{"role":"system", "content":system_prompt}, {"role":"user", "content":scenario_text}]
    assistant_history = []
    latencies = []
    tool_called = False
    tool_json = None
    tool_parse_err = None
    tool_response = None
    turn_count = 0

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
            
            if dynamic_trigger and not dynamic_triggered and turn_count >= dynamic_trigger["turn"]:
                tool_parse_err = "premature_commitment"
                messages.append({"role":"assistant", "content":assistant_text})
                msg = "Wait, don't book yet! Cancel that if you did. " + dynamic_trigger["injection"]
                messages.append({"role":"user", "content":msg})
                current_expected = dynamic_trigger.get("new_expected", current_expected)
                dynamic_triggered = True
                continue
            elif dynamic_trigger and turn_count < dynamic_trigger["turn"]:
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
            
            if dynamic_trigger and not dynamic_triggered and turn_count >= dynamic_trigger["turn"]:
                messages.append({"role":"user", "content":dynamic_trigger["injection"]})
                current_expected = dynamic_trigger.get("new_expected", current_expected)
                dynamic_triggered = True
                continue

            if re.search(r'\b(how many|party size|what time|when is|what city|which cuisine)\b', assistant_text, flags=re.IGNORECASE):
                if re.search(r'\b(how many|party size|how many people|number of people)\b', assistant_text.lower()):
                    user_reply = f"{current_expected.get('people')}"
                elif re.search(r'\b(what time|when is|what date|which day)\b', assistant_text.lower()):
                    user_reply = f"{current_expected.get('day') or current_expected.get('time') or ''}"
                elif re.search(r'\b(what city|which city|where)\b', assistant_text.lower()):
                    user_reply = current_expected.get("city", "")
                elif re.search(r'\b(cuisine|what kind of|which cuisine)\b', assistant_text.lower()):
                    user_reply = current_expected.get("cuisine", "")
                else:
                    user_reply = None
                if user_reply:
                    messages.append({"role":"user", "content":user_reply})
                    continue

            if re.search(r'\b(reservation confirmed|booking confirmed|i have booked|i booked|confirmation id)\b', assistant_text, flags=re.IGNORECASE):
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

    final_text = "\n\n".join(assistant_history) if assistant_history else ""

    return {
        "final_text": final_text,
        "assistant_history": assistant_history,
        "latencies": latencies,
        "tool_called": tool_called,
        "tool_json": tool_json,
        "tool_parse_err": tool_parse_err,
        "tool_response": tool_response,
        "turn_count": turn_count
    }
def create_publication_figures(df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: str):
    sns.set_style("whitegrid")
    sns.set_palette("husl")

    # Figure 1: Main Comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    model_groups = summary_df.groupby("model").agg({
        "mean_score": "mean",
        "ci95_lower": "min",
        "ci95_upper": "max"
    }).reset_index()

    models = model_groups["model"].tolist()
    means = model_groups["mean_score"].tolist()
    errors_lower = [means[i] - model_groups.iloc[i]["ci95_lower"] for i in range(len(models))]
    errors_upper = [model_groups.iloc[i]["ci95_upper"] - means[i] for i in range(len(models))]

    colors = sns.color_palette("husl", len(models))

    bars = ax.bar(range(len(models)), means,
                   yerr=[errors_lower, errors_upper],
                   capsize=8, alpha=0.8, color=colors,
                   error_kw={"linewidth": 2.5, "ecolor": "black"})

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean Score (0-20)", fontsize=13, fontweight="bold")
    ax.set_title("TrustTrajectory: Model Performance Comparison",
                 fontsize=14, fontweight="bold", pad=20)
    ax.set_ylim([0, 22])
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    for i, (bar, mean) in enumerate(zip(bars, means)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f"{mean:.1f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.axhline(y=15, color="red", linestyle="--", linewidth=2, alpha=0.5, label="Good Performance (15/20)")
    ax.axhline(y=10, color="orange", linestyle="--", linewidth=2, alpha=0.5, label="Acceptable (10/20)")

    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig1_main_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Figure 1 saved: fig1_main_comparison.png")

    # Figure 2: Dimension Breakdown
    fig, ax = plt.subplots(figsize=(12, 6))

    dimension_data = df.groupby("model")[["constraint_fidelity", "hallucination_resilience", "quality_score"]].mean()

    x = range(len(dimension_data.index))
    width = 0.5

    p1 = ax.bar(x, dimension_data["constraint_fidelity"], width, label="Constraint Fidelity", color="#FF6B6B")
    p2 = ax.bar(x, dimension_data["hallucination_resilience"], width, bottom=dimension_data["constraint_fidelity"],
                label="Hallucination Resilience", color="#4ECDC4")
    p3 = ax.bar(x, dimension_data["quality_score"], width,
                bottom=dimension_data["constraint_fidelity"] + dimension_data["hallucination_resilience"],
                label="Response Quality", color="#FFA502")

    ax.set_xticks(x)
    ax.set_xticklabels(dimension_data.index, fontsize=12, fontweight="bold")
    ax.set_ylabel("Average Score per Dimension", fontsize=13, fontweight="bold")
    ax.set_title("TrustTrajectory: Performance Breakdown by Evaluation Dimension",
                 fontsize=14, fontweight="bold", pad=20)
    ax.set_ylim([0, 22])
    ax.legend(loc="upper right", fontsize=11, frameon=True, shadow=True)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig2_dimension_breakdown.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Figure 2 saved: fig2_dimension_breakdown.png")

    fig, ax = plt.subplots(figsize=(12, 6))

    pass_data = summary_df.groupby("model").agg({
        "pass_at_10": "mean",
        "pass_at_15": "mean"
    }).reset_index()

    x = range(len(pass_data))
    width = 0.35

    bars1 = ax.bar([i - width/2 for i in x], pass_data["pass_at_10"], width,
                    label="pass@10 (Acceptable)", color="#3498db", alpha=0.8)
    bars2 = ax.bar([i + width/2 for i in x], pass_data["pass_at_15"], width,
                    label="pass@15 (Good)", color="#e74c3c", alpha=0.8)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f"{height:.2f}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(pass_data["model"], fontsize=12, fontweight="bold")
    ax.set_ylabel("Consistency (Proportion)", fontsize=13, fontweight="bold")
    ax.set_title("TrustTrajectory: Model Reliability via pass@k Metrics",
                 fontsize=14, fontweight="bold", pad=20)
    ax.set_ylim([0, 1.1])
    ax.legend(loc="upper right", fontsize=11, frameon=True, shadow=True)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig3_pass_at_k.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Figure 3 saved: fig3_pass_at_k.png")

def run_benchmark(client_openai, client_anthropic, models: List[str], scenarios: List[Dict[str,Any]],
                  n_runs: int, temperature: float, max_tokens: int, max_turns: int, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    timestamp = now_iso()
    all_rows = []
    summary = {}
    trajectory_data = {}

    print(f"\n{'='*70}")
    print(f"Running Benchmark: {BENCHMARK_NAME}")
    print(f"Version: {BENCHMARK_VERSION} | Models: {len(models)} | Runs: {n_runs}")
    print(f"Random Seed: {seed}")
    print(f"{'='*70}\n")

    for model in models:
        print(f"\n--- Model: {model} ---")
        for run_idx in range(1, n_runs + 1):
            print(f"  Run {run_idx}/{n_runs}")
            for scen in scenarios:
                sid = scen.get("id", str(uuid.uuid4())[:8])
                text = scen["text"]
                expected = scen.get("expected", {})

                res = run_single_scenario(client_openai, client_anthropic, model, text, expected, temperature,
                                         max_tokens, max_turns, run_idx, seed)

                if res.get("error"):
                    print(f"    ⚠ Scenario {sid}: {res['error']}")
                    # Create fallback row with zero scores to preserve data structure
                    row = {
                        "timestamp": timestamp,
                        "model": model,
                        "run_idx": run_idx,
                        "scenario_id": sid,
                        "scenario_text": text,
                        "expected": json.dumps(expected, ensure_ascii=False),
                        "final_text": f"[ERROR: {res['error']}]",
                        "total_score": 0,
                        "constraint_fidelity": 0,
                        "hallucination_resilience": 0,
                        "quality_score": 0,
                        "tool_called": False,
                        "tool_json": "",
                        "tool_parse_err": "api_error",
                        "tool_response": "",
                        "latency_mean_s": None,
                        "turn_count": res.get("turn_failed", 0),
                        "n_api_calls": 0,
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    }
                    all_rows.append(row)
                    continue

                final_text = res["final_text"]
                tool_flag = res["tool_called"]
                tool_json = res["tool_json"]
                tool_resp = res["tool_response"]
                latencies = res["latencies"]
                turn_count = res["turn_count"]
                latency_mean = statistics.mean(latencies) if latencies else None

                total_score, breakdown = score_response(final_text, expected, tool_flag, tool_resp, res.get("tool_parse_err"))

                row = {
                    "timestamp": timestamp,
                    "model": model,
                    "run_idx": run_idx,
                    "scenario_id": sid,
                    "scenario_text": text,
                    "expected": json.dumps(expected, ensure_ascii=False),
                    "final_text": final_text,
                    "total_score": total_score,
                    "constraint_fidelity": breakdown["constraint_fidelity"],
                    "hallucination_resilience": breakdown["hallucination_resilience"],
                    "quality_score": breakdown["quality"],
                    "tool_called": tool_flag,
                    "tool_json": json.dumps(tool_json, ensure_ascii=False) if tool_json else "",
                    "tool_parse_err": res.get("tool_parse_err"),
                    "tool_response": json.dumps(tool_resp, ensure_ascii=False) if tool_resp else "",
                    "latency_mean_s": latency_mean,
                    "turn_count": turn_count,
                    "n_api_calls": len(latencies),
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                all_rows.append(row)

                key = (model, sid)
                summary.setdefault(key, []).append(total_score)

                traj_quality = calculate_trajectory_quality(res["assistant_history"])
                if key not in trajectory_data:
                    trajectory_data[key] = []
                trajectory_data[key].append(traj_quality)

                print(f"    Scenario {sid} | Score {total_score}/20 | Turns: {turn_count} | Tool: {tool_flag} | Latency: {latency_mean:.2f}s")

    #Save results
    df = pd.DataFrame(all_rows)
    detailed_csv = os.path.join(OUTPUT_DIR, "detailed_results.csv")
    df.to_csv(detailed_csv, index=False)
    print(f"\n✓ Detailed results: {detailed_csv}")

    #Generate summary
    summary_rows = []
    for (model, sid), scores in summary.items():
        mean, ci95, se = mean_ci95(scores)
        stdev = statistics.stdev(scores) if len(scores) > 1 else 0.0

        pass_at_15 = calculate_pass_at_k(scores, threshold=15)
        pass_at_10 = calculate_pass_at_k(scores, threshold=10)

        summary_rows.append({
            "model": model,
            "scenario_id": sid,
            "n_runs": len(scores),
            "mean_score": round(mean, 2),
            "std_dev": round(stdev, 2),
            "se": round(se, 2),
            "ci95_lower": round(mean - ci95, 2),
            "ci95_upper": round(mean + ci95, 2),
            "pass_at_10": round(pass_at_10, 3),
            "pass_at_15": round(pass_at_15, 3),
            "min": min(scores),
            "max": max(scores)
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(OUTPUT_DIR, "summary_results.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"✓ Summary statistics: {summary_csv}")

    #Save metadata
    metadata = {
        "benchmark_name": BENCHMARK_NAME,
        "benchmark_version": BENCHMARK_VERSION,
        "research_focus": RESEARCH_FOCUS,
        "generated_at": timestamp,
        "random_seed": seed,
        "n_runs_per_scenario": n_runs,
        "models_tested": models,
        "n_scenarios": len(scenarios),
        "total_runs": len(all_rows),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "max_turns": max_turns
    }
    save_json(metadata, os.path.join(OUTPUT_DIR, "metadata.json"))
    print(f"✓ Metadata: metadata.json")

    # Save trajectory analysis
    trajectory_summary = {}
    for (model, sid), trajs in trajectory_data.items():
        if not trajs:
            continue
        avg_turns = round(statistics.mean([t["trajectory_turns"] for t in trajs]), 2)
        avg_tool_attempts = round(statistics.mean([t["tool_attempts"] for t in trajs]), 2)
        avg_clarifications = round(statistics.mean([t["clarification_turns"] for t in trajs]), 2)
        trajectory_summary[f"{model}_{sid}"] = {
            "avg_turns": avg_turns,
            "avg_tool_attempts": avg_tool_attempts,
            "avg_clarifications": avg_clarifications
        }
    save_json(trajectory_summary, os.path.join(OUTPUT_DIR, "trajectory_analysis.json"))
    print(f"✓ Trajectory analysis: trajectory_analysis.json")

    #visualizations
    print("\nGenerating figures")
    if not df.empty and not summary_df.empty:
        try:
            create_publication_figures(df, summary_df, OUTPUT_DIR)
        except Exception as e:
            print(f"⚠ Visualization error: {e}")
            print("  (Data collection successful - figures are optional)")

    #LaTeX table
    latex_file = os.path.join(OUTPUT_DIR, "results_table.txt")
    with open(latex_file, "w", encoding="utf-8") as f:
        f.write("% RESEARCH-GRADE RESULTS TABLE WITH pass@k METRIC\n")
        f.write("% Include in paper with: \\input{results_table.txt}\n\n")
        f.write("\\begin{table}[h]\n\\centering\n\\small\n")
        f.write("\\begin{tabular}{l l r r r r r}\n")
        f.write("\\hline\n")
        f.write("Model & Scenario & $n$ & Mean & \\textbf{95\\% CI} & pass@10 & pass@15 \\\\\n")
        f.write("\\hline\n")
        for r in summary_rows:
            ci_str = f"[{r['ci95_lower']}, {r['ci95_upper']}]"
            f.write(f"{r['model']} & {r['scenario_id']} & {r['n_runs']} & {r['mean_score']} & {ci_str} & {r['pass_at_10']} & {r['pass_at_15']} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")
        f.write("\\caption{Long-Trajectory Agentic Execution benchmark. pass@k measures consistency.}\n")
        f.write("\\label{tab:lta-e-results}\n\\end{table}\n")
    print(f"✓ LaTeX table: {latex_file}")

    return {
        "detailed_csv": detailed_csv,
        "summary_csv": summary_csv,
        "metadata": os.path.join(OUTPUT_DIR, "metadata.json"),
        "trajectory_analysis": os.path.join(OUTPUT_DIR, "trajectory_analysis.json"),
        "latex_table": latex_file
    }

#Setup OpenAI API
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    openai_api_key = input("Enter your OpenAI API key (or press Enter to skip): ").strip()
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key

#Anthropic API SEtup
anthropic_api_key = None
needs_anthropic = any("claude" in model.lower() for model in DEFAULT_MODELS)

if needs_anthropic:
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        anthropic_api_key = input("Enter your Anthropic API key (for Claude models): ").strip()
        if anthropic_api_key:
            os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key

#Initialize clients
client_openai = OpenAI(api_key=openai_api_key) if (openai_api_key and OpenAI) else None
client_anthropic = Anthropic(api_key=anthropic_api_key) if (anthropic_api_key and Anthropic) else None

#Validate clients
openai_models = [m for m in DEFAULT_MODELS if "claude" not in m.lower()]
claude_models = [m for m in DEFAULT_MODELS if "claude" in m.lower()]

if openai_models and not client_openai:
    raise ValueError(f"OpenAI models requested ({openai_models}) but no API key provided")
if claude_models and not client_anthropic:
    raise ValueError(f"Claude models requested ({claude_models}) but no Anthropic API key provided")

#Runbenchmark
results = run_benchmark(
    client_openai=client_openai,
    client_anthropic=client_anthropic,
    models=DEFAULT_MODELS,
    scenarios=SCENARIOS,
    n_runs=N_RUNS,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
    max_turns=MAX_TURNS,
    seed=RANDOM_SEED
)

print(f"\n{'='*70}")
print("BENCHMARK COMPLETE")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Files generated: {results}")
print(f"{'='*70}")