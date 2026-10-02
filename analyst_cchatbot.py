prompt = ChatPromptTemplate.
from_messages([
 ("system", SYSTEM_PROMPT), ("human", "{question}") ])