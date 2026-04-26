
#------------------ current flow (fast api) -------------------------
1. Agent already knows the endpoints (hardcoded or from README)
2. It sends request → /tools/weather
3. Server receives it
4. Server calls the weather logic
5. Returns JSON response

Agent → fixed URL → server → tool → response

problem:Agent is “blind” unless you manually tell it what exists
Not flexible or scalable


#--------------- mcp flow -----------------
1. Agent first asks: “What tools do you have?” → /tools
2. Server responds with a list of tools + how to use them
3. Agent chooses a tool (e.g., weather)
Sends a standard request (JSON-RPC)
Server runs the tool and responds
4. Agent optionally calls /step to move simulation forward

Agent → asks capabilities → server explains → agent decides → tool runs → response