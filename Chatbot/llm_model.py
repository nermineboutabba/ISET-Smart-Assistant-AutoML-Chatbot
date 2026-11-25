import os

_llm_pipeline = None
_llm_attempted = False


def _get_llm():
    """Lazy-load optional HuggingFace text generation pipeline when enabled."""
    global _llm_pipeline, _llm_attempted
    if _llm_attempted:
        return _llm_pipeline
    _llm_attempted = True

    if os.environ.get("ENABLE_LOCAL_LLM", "0") != "1":
        return None

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        import torch

        model_name = os.environ.get("LLM_MODEL_NAME", "EleutherAI/gpt-neo-1.3B")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        _llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
    except Exception as exc:
        print(f"[llm_model] Falling back to retrieved FAQ answer ({exc})")
        _llm_pipeline = None
    return _llm_pipeline


def generate_answer(user_question, matched_question, matched_answer, source=None):
    """Synthesize a concise ISET Sfax reply, falling back safely offline."""
    llm = _get_llm()
    if llm is None:
        return matched_answer

    prompt = (
        "You are an assistant for ISET Sfax. Answer strictly using the matched FAQ snippet below.\n\n"
        f'User Question: "{user_question}"\n'
        f'Matched Question: "{matched_question}"\n'
        f'Matched Answer: "{matched_answer}"\n'
        "Final Answer:\n"
    )
    tokenizer = llm.tokenizer
    output = llm(
        prompt,
        max_new_tokens=80,
        temperature=0.3,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )[0]["generated_text"]

    if "Final Answer:" in output:
        return output.split("Final Answer:")[-1].strip()
    return output.strip() or matched_answer
