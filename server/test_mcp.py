# --- test case 1:---------------------

# import requests

# url = "http://localhost:7860/mcp"

# headers = {
#     "Content-Type": "application/json",
#     "Accept": "text/event-stream"
# }

# data = {
#     "type": "tools/list"
# }

# response = requests.post(url, headers=headers, json=data, stream=True)

# for line in response.iter_lines():
#     if line:
#         print(line.decode())
        
# test case result: {"jsonrpc":"2.0","id":"server-error","error":{"code":-32600,"message":"Not Acceptable: Client must accept both application/json and text/event-stream"}}
# fastmcp expect both Accept: application/json, text/event-stream not only one.

#------------------- test case 2 -------------------------

# import requests

# url = "http://localhost:7860/mcp"

# headers = {
#     "Content-Type": "application/json",
#     "Accept": "application/json, text/event-stream"
# }

# data = {
#     "type": "tools/list"
# }

# response = requests.post(url, headers=headers, json=data, stream=True)

# for line in response.iter_lines():
#     if line:
#         print(line.decode())
# # ------------------ conslusion: But FastMCP expects JSON-RPC 2.0 format, not plain JSON. 


#----------------------- test case 3 : ----------------------------

# import requests

# url = "http://localhost:7860/mcp"

# headers = {
#     "Content-Type": "application/json",
#     "Accept": "application/json, text/event-stream"
# }

# data = {
#     "jsonrpc": "2.0",
#     "id": 1,
#     "method": "tools/list"
# }

# response = requests.post(url, headers=headers, json=data, stream=True)

# print("STATUS:", response.status_code)
# print("HEADERS:", response.headers)

# for line in response.iter_lines(decode_unicode=True):
#     print("RAW:", line)


# ----------------------- test case 4 --------------------------------
# import requests

# url = "http://localhost:7860/mcp"

# headers = {
#     "Content-Type": "application/json",
#     "Accept": "application/json, text/event-stream"
# }

# # -----------------------------
# # STEP 1: Initialize session
# # -----------------------------
# init_data = {
#     "jsonrpc": "2.0",
#     "id": 1,
#     "method": "initialize"
# }

# init_res = requests.post(url, headers=headers, json=init_data)

# print("INIT STATUS:", init_res.status_code)

# session_id = init_res.headers.get("mcp-session-id")
# print("SESSION ID:", session_id)

# # Add session header
# headers["mcp-session-id"] = session_id


# # -----------------------------
# # STEP 2: List tools
# # -----------------------------
# list_data = {
#     "jsonrpc": "2.0",
#     "id": 2,
#     "method": "tools/list"
# }

# res = requests.post(url, headers=headers, json=list_data)

# print("TOOLS:", res.text)


# # -----------------------------
# # STEP 3: Call weather tool
# # -----------------------------
# call_data = {
#     "jsonrpc": "2.0",
#     "id": 3,
#     "method": "tools/call",
#     "params": {
#         "name": "weather",
#         "arguments": {
#             "days_ahead": 3
#         }
#     }
# }

# res = requests.post(url, headers=headers, json=call_data)

# print("WEATHER:", res.text)

# #-------------output : 
# INIT STATUS: 200
# SESSION ID: b07d18015b9a48bf8295b547fae3dc0c
# TOOLS: event: message
# data: {"jsonrpc":"2.0","id":2,"error":{"code":-32602,"message":"Invalid request parameters","data":""}}


# WEATHER: event: message
# data: {"jsonrpc":"2.0","id":3,"error":{"code":-32602,"message":"Invalid request parameters","data":""}}


#-------------------------------- PS E:\meta_hack\meta-hackathon-final-idea\server> 
# import requests

# url = "http://localhost:7860/mcp"

# headers = {
#     "Content-Type": "application/json",
#     "Accept": "application/json, text/event-stream"
# }

# # STEP 1: INIT
# init_data = {
#     "jsonrpc": "2.0",
#     "id": 1,
#     "method": "initialize",
#     "params": {}
# }

# init_res = requests.post(url, headers=headers, json=init_data)

# session_id = init_res.headers.get("mcp-session-id")
# headers["mcp-session-id"] = session_id

# print("SESSION:", session_id)


# # STEP 2: LIST TOOLS
# list_data = {
#     "jsonrpc": "2.0",
#     "id": 2,
#     "method": "tools/list",
#     "params": {}
# }

# res = requests.post(url, headers=headers, json=list_data, stream=True)

# for line in res.iter_lines(decode_unicode=True):
#     if line and line.startswith("data:"):
#         print("TOOLS:", line.replace("data: ", ""))


# # STEP 3: CALL TOOL
# call_data = {
#     "jsonrpc": "2.0",
#     "id": 3,
#     "method": "tools/call",
#     "params": {
#         "tool_name": "weather",
#         "arguments": {
#             "days_ahead": 3
#         }
#     }
# }

# res = requests.post(url, headers=headers, json=call_data, stream=True)

# for line in res.iter_lines(decode_unicode=True):
#     if line and line.startswith("data:"):
#         print("WEATHER:", line.replace("data: ", ""))

#------------------------output: it is not satsifying the parameter formate , so it is througing invalid params ------------------------------
import requests

url = "http://localhost:7860/mcp"
headers = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream"
}

# 1. Initialize with proper params
init_res = requests.post(url, headers=headers, json={
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "test-client", "version": "1.0.0"}
    }
})
session_id = init_res.headers.get("mcp-session-id")
headers["mcp-session-id"] = session_id
print("SESSION:", session_id)
print("INIT:", init_res.text)

# 2. Send initialized notification (required handshake)
requests.post(url, headers=headers, json={
    "jsonrpc": "2.0",
    "method": "notifications/initialized"
    # no "id" — this is a notification, not a request
})

# 3. List tools (params omitted)
tools_res = requests.post(url, headers=headers, json={
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/list"
})
print("\nTOOLS:", tools_res.text)

# 4. Call weather tool
weather_res = requests.post(url, headers=headers, json={
    "jsonrpc": "2.0",
    "id": 3,
    "method": "tools/call",
    "params": {
        "name": "weather",
        "arguments": {"days_ahead": 3}
    }
})
print("\nWEATHER:", weather_res.text)