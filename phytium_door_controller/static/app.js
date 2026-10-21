/*
前端控制脚本：只调用 /api/*，不直接处理硬件逻辑。
状态刷新、按钮命令、AI 指令都集中在这里，方便初学者调试。
*/

function $(id) {
  return document.getElementById(id);
}

async function postJson(url, body) {
  // 可选控制 token：模拟模式默认空；正式/局域网部署时在浏览器 localStorage 设置 door_token。
  const token = localStorage.getItem("door_token") || "";
  const headers = { "Content-Type": "application/json" };
  if (token) {
    headers["X-Door-Token"] = token;
  }
  const resp = await fetch(url, {
    method: "POST",
    headers: headers,
    body: JSON.stringify(body || {})
  });
  return resp.json();
}

async function refreshStatus() {
  try {
    const resp = await fetch("/api/status");
    const data = await resp.json();
    $("state").textContent = data.state;
    $("backend").textContent = data.backend;
    $("limit-open").textContent = data.limits.open ? "触发" : "未触发";
    $("limit-close").textContent = data.limits.close ? "触发" : "未触发";
    $("estop").textContent = data.emergency_stop ? "触发" : "未触发";
    $("logs").textContent = (data.logs || []).join("\n");
  } catch (err) {
    $("state").textContent = "web_api_error";
  }
}

function bindButton(id, url) {
  $(id).addEventListener("click", async () => {
    await postJson(url, {});
    refreshStatus();
  });
}

bindButton("btn-open", "/api/open");
bindButton("btn-close", "/api/close");
bindButton("btn-stop", "/api/stop");
bindButton("btn-reset", "/api/reset");

$("btn-ai").addEventListener("click", async () => {
  const text = $("ai-text").value;
  const result = await postJson("/api/ai-command", { text: text });
  $("ai-result").textContent = JSON.stringify(result, null, 2);
  refreshStatus();
});

refreshStatus();
setInterval(refreshStatus, 1000);
