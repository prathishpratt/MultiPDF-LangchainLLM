import os
from langchain import PromptTemplate, HuggingFaceHub
from langchain import LLMChain

#Give your api token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

prompt_template = "Question: Tell me who won the {sport} in {year}"
llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length": 1000})

llm_chain = LLMChain(prompt=PromptTemplate.from_template(prompt_template), llm=llm)

print(llm_chain.run({"sport": "ICC criket world cup","year":"2012"}))