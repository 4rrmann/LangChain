from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()
result = search.invoke("AOT")

# print(result)



from langchain_community.tools import ShellTool

shelltool = ShellTool()
result = shelltool.invoke("whoami")

print(result)