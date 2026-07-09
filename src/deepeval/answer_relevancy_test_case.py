from deepeval import evaluate
from langchain_ollama import ChatOllama
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase

execution_model = ChatOllama(model="jeffnyman/ts-reasoner")
judge_model = OllamaModel(model="jeffnyman/ts-evaluator")

metric = AnswerRelevancyMetric(model=judge_model, verbose_mode=True)

question = "What does the Higgs boson explain in particle physics?"

response = execution_model.invoke(question).content

generated = LLMTestCase(
  input=question,
  actual_output = response
)

# To run metric as a standalone
#metric.measure(generated)

evaluate(test_cases=[generated], metrics=[metric])