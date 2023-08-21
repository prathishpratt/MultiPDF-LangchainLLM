#Just running the test to check how langchain works
from langchain.llms import OpenAI

llm = OpenAI(openai_api_key="sk-HeG...") #Input yours

llm("Where was 2012 Olymbics held?")
llm("Who is captained India for 2012 world cup?")
