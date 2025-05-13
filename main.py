from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import uuid
import os
import time
import random
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HIGH_LATENCY_MODELS = os.getenv("HIGH_LATENCY_MODELS", "test-model").split(",")
ERROR_MODELS = os.getenv("ERROR_MODELS", "test-model").split(",")

NORMAL_MIN_LATENCY = float(os.getenv("NORMAL_MIN_LATENCY", "0.3"))
NORMAL_MAX_LATENCY = float(os.getenv("NORMAL_MAX_LATENCY", "1.0"))
HIGH_MIN_LATENCY = float(os.getenv("HIGH_MIN_LATENCY", "3.0"))
HIGH_MAX_LATENCY = float(os.getenv("HIGH_MAX_LATENCY", "5.0"))

def should_increase_latency(model):
    if model not in HIGH_LATENCY_MODELS:
        return False
        
    current_minute = datetime.now().minute
    return 15 <= current_minute < 30

def should_return_401(model):
    if model not in ERROR_MODELS:
        return False
        
    current_minute = datetime.now().minute
    return 45 <= current_minute < 55

def get_tokens_for_response():
    opening_phrases = [
        "I've been considering this topic for a while.",
        "Let me share my thoughts on this matter.",
        "There are several important aspects to consider.",
        "This is a fascinating subject with many dimensions.",
        "Here's what I think about this question.",
        "I'd like to approach this from multiple angles.",
        "When analyzing this problem, several factors emerge.",
        "My perspective on this has evolved over time.",
        "The answer isn't straightforward, but I'll explain.",
        "Let me break this down step by step.",
    ]
    
    technical_terms = [
        "neural networks", "backpropagation", "gradient descent", "regularization",
        "convolutional layers", "recurrent architectures", "transformers", "attention mechanisms",
        "embeddings", "tokenization", "fine-tuning", "transfer learning", 
        "supervised learning", "unsupervised methods", "reinforcement learning",
        "generative models", "discriminative algorithms", "latent space", "vector quantization",
    ]
    
    abstract_concepts = [
        "information theory", "cognitive bias", "emergence", "complexity theory",
        "computational linguistics", "statistical inference", "decision theory",
        "game theory", "probabilistic modeling", "causal reasoning", "bayesian networks",
        "evolutionary computation", "swarm intelligence", "metaheuristics",
    ]
    
    connecting_phrases = [
        "furthermore", "moreover", "in addition", "consequently", "however",
        "nevertheless", "therefore", "thus", "in contrast", "on the other hand",
        "similarly", "specifically", "particularly", "notably", "in particular",
        "to illustrate this", "as an example", "to put it differently",
    ]
    
    ending_sentences = [
        "This remains an active area of research.",
        "Further developments will likely emerge in this field.",
        "The implications of this are still being explored.",
        "These insights continue to evolve with new discoveries.",
        "This perspective offers valuable context for future work.",
        "The full potential of these approaches is yet to be realized.",
        "This framework provides a foundation for deeper analysis.",
        "These considerations shape our understanding of the domain.",
        "The interplay of these factors drives ongoing innovation.",
        "Such nuanced understanding helps advance the state of the art.",
    ]

    num_sentences = random.randint(5, 12)
    sentences = []
    
    sentences.append(random.choice(opening_phrases))
    
    for i in range(num_sentences - 2): 
        sentence_type = random.randint(1, 5)
        
        if sentence_type == 1:
            term1 = random.choice(technical_terms)
            term2 = random.choice(technical_terms)
            sentences.append(f"The relationship between {term1} and {term2} demonstrates important principles that influence how we approach problem-solving in this domain.")
        
        elif sentence_type == 2:
            concept = random.choice(abstract_concepts)
            term = random.choice(technical_terms)
            sentences.append(f"When we examine {concept}, we find parallels to {term} that illuminate underlying patterns in complex systems.")
        
        elif sentence_type == 3:
            term1 = random.choice(technical_terms)
            term2 = random.choice(technical_terms)
            connector = random.choice(connecting_phrases)
            sentences.append(f"While {term1} offers certain advantages in efficiency, {connector}, {term2} provides greater flexibility in diverse applications.")
        
        elif sentence_type == 4:
            concept = random.choice(abstract_concepts)
            connector = random.choice(connecting_phrases)
            sentences.append(f"{connector.capitalize()}, {concept} plays a crucial role in shaping our understanding of emergent properties in dynamic systems.")
        
        else:
            term = random.choice(technical_terms + abstract_concepts)
            sentences.append(f"The field of {term} has undergone significant transformation in recent years due to advances in computational capabilities and methodological approaches.")
    
    sentences.append(random.choice(ending_sentences))
    
    estimated_tokens = sum(len(s.split()) * 1.3 for s in sentences)  
    while estimated_tokens > 200 and len(sentences) > 5:
        remove_idx = random.randint(1, len(sentences) - 2)
        estimated_tokens -= len(sentences[remove_idx].split()) * 1.3
        sentences.pop(remove_idx)
    
    filler_phrases = [
        f"This is particularly evident when considering {random.choice(technical_terms)}.",
        f"The implications extend to various domains including {random.choice(abstract_concepts)}.",
        f"We can observe similar patterns in {random.choice(technical_terms)} and {random.choice(technical_terms)}.",
        f"Many researchers have explored this connection between theory and application.",
        f"The evolution of these ideas reflects broader trends in the field.",
    ]
    
    while estimated_tokens < 100 and filler_phrases:
        insert_idx = random.randint(1, len(sentences) - 1)
        filler = filler_phrases.pop(random.randint(0, len(filler_phrases) - 1))
        sentences.insert(insert_idx, filler)
        estimated_tokens += len(filler.split()) * 1.3
    
    result = []
    for i, sentence in enumerate(sentences):
        if i > 0 and random.random() < 0.3:  # 30% chance to add a connector
            result.append(random.choice(connecting_phrases).capitalize() + ",")
        result.append(sentence)
    
    return " ".join(result)

def streaming_data_generator(content):
    """Generate a streaming response with the given content"""
    response_id = uuid.uuid4().hex
    words = content.split(" ")
    for word in words:
        word = word + " "
        chunk = {
            "id": f"chatcmpl-{response_id}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "gpt-3.5-turbo",
            "choices": [{"index": 0, "delta": {"content": word}}],
        }
        try:
            yield f"data: {json.dumps(chunk)}\n\n"
        except:
            yield f"data: {json.dumps(chunk)}\n\n"
    
    yield f"data: [DONE]\n\n"

# Chat completions endpoint supporting both OpenAI and Azure formats
@app.post("/chat/completions")
@app.post("/v1/chat/completions")
@app.post("/openai/deployments/{model:path}/chat/completions")  # azure compatible endpoint
async def completion(request: Request, model: str = None, authorization: str = Header(default=None)):
    data = await request.json()
    requested_model = data.get("model") or model or "gpt-3.5-turbo"

    if(requested_model == "no-latency-model"):
        response_id = uuid.uuid4().hex
        response = {
            "id": f"chatcmpl-{response_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": requested_model,
            "system_fingerprint": "fp_" + uuid.uuid4().hex[:12],
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello, how are you?",
                    },
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": random.randint(10, 50),
                "completion_tokens": 10,
                "total_tokens": 20
            },
        }
        print(f"Non-streaming response for {requested_model}")
        return response
    
    if should_return_401(requested_model):
        print(f"Returning 401 error for model {requested_model} - current time: {datetime.now().strftime('%H:%M')}")
        raise HTTPException(
            status_code=401, 
            detail="Unauthorized: Invalid Authentication"
        )
    
    if should_increase_latency(requested_model):
        latency = random.uniform(HIGH_MIN_LATENCY, HIGH_MAX_LATENCY)
        print(f"Increased latency for model {requested_model}: {latency:.2f}s - current time: {datetime.now().strftime('%H:%M')}")
    else:
        latency = random.uniform(NORMAL_MIN_LATENCY, NORMAL_MAX_LATENCY)
        print(f"Normal latency for model {requested_model}: {latency:.2f}s - current time: {datetime.now().strftime('%H:%M')}")
    
    await asyncio.sleep(latency)
    
    content = get_tokens_for_response()
    
    if data.get("stream") == True:
        print(f"Streaming response for {requested_model}")
        return StreamingResponse(
            content=streaming_data_generator(content),
            media_type="text/event-stream",
        )
    else:
        response_id = uuid.uuid4().hex
        response = {
            "id": f"chatcmpl-{response_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": requested_model,
            "system_fingerprint": "fp_" + uuid.uuid4().hex[:12],
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": random.randint(10, 50),
                "completion_tokens": len(content.split()),
                "total_tokens": random.randint(10, 50) + len(content.split())
            },
        }
        print(f"Non-streaming response for {requested_model}")
        return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8090)
