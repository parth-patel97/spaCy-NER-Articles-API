# Python 3.10.6

from fastapi import FastAPI
from ml import NER
from pydantic import BaseModel
from typing import List


app = FastAPI()

@app.get("/")
async def hello():
    '''
    Simple GET API
    '''
    return "Hello World!"

# Create a model for Articles
class Article(BaseModel):
    content:str
    comments: List[str] = []


@app.post("/article/")
async def analyze_article(articles:List[Article]):
    '''
    Analyze an article and extract entities from it using Spacy 
    '''
    """
    request body: [
    {
        "content": "Apple buys U.K. startup for $1 billion dollars",
        "comments": ["About money","its good"]
    }]

    Response body :
            {
        "ents": [
            {
                "text": "Apple",
                "label": "ORG"
            },
            {
                "text": "U.K.",
                "label": "GPE"
            },
            {
                "text": "$1 billion dollars",
                "label": "MONEY"
            }
        ],
        "comments": [
                "ABOUT MONEY",
                "ITS GOOD"
        ] } 
    
    """
    ents = []
    comments = []
    for article in articles:
        for comment in article.comments:
            comments.append(comment.upper())
        doc = NER(article.content)
        for ent in doc.ents:
            ents.append({"text":ent.text,"label":ent.label_})
        
    return {"ents":ents,"comments":comments}
