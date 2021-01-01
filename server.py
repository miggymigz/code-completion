from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ccompletion.samplers import sampleGPT2, sampleT5
from transformers import (
    GPT2LMHeadModel, GPT2TokenizerFast,
    T5ForConditionalGeneration, T5TokenizerFast,
)

objects = {}
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SamplePayload(BaseModel):
    src: str


@app.on_event('startup')
def startup_event():
    # initialize GPT-2 model
    gpt2model = GPT2LMHeadModel.from_pretrained('checkpoints/gpt2')
    gpt2tokenizer = GPT2TokenizerFast.from_pretrained('gpt2-medium')
    print('DEBUG: done initializing GPT-2')

    # initialize T5 model
    t5model = T5ForConditionalGeneration.from_pretrained('checkpoints/t5')
    t5tokenizer = T5TokenizerFast.from_pretrained('t5-base')
    print('DEBUG: done initializing T5')

    # store models and tokenizers for subsequent requests
    objects['gpt2'] = gpt2model, gpt2tokenizer
    objects['t5'] = t5model, t5tokenizer


@app.post('/api/sampleGPT2')
def doSampleGPT2(payload: SamplePayload):
    src = payload.src
    model, tokenizer = objects['gpt2']
    results = sampleGPT2(model=model, tokenizer=tokenizer, sequence=src)
    return [{'probability': p, 'value': v} for p, v in results]


@app.post('/api/sampleT5')
def doSampleT5(payload: SamplePayload):
    src = payload.src
    model, tokenizer = objects['t5']

    try:
        results = sampleT5(model=model, tokenizer=tokenizer, sequence=src)
    except ValueError as e:
        return HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )

    return [{'probability': 0, 'value': value} for value in results]
