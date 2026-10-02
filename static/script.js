const $ = (id) => document.getElementById(id);
const fmt = (n) => Number(n).toLocaleString();

async function api(url, opts) {
  const res = await fetch(url, opts);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || "Request failed");
  return data;
}

function renderStats(s) {
  $("c-total").textContent = fmt(s.total);
  $("c-normal").textContent = fmt(s.normal);
  $("c-attack").textContent = fmt(s.attack);
  const share = s.total ? (s.attack / s.total) * 100 : 0;
  $("c-share").textContent = share.toFixed(1) + "%";

  const donut = $("donut");
  if (!s.total) {
    donut.style.background = "rgba(255,255,255,.1)";
    donut.dataset.label = "No data yet";
  } else {
    const a = (s.normal / s.total) * 360;
    donut.style.background = `conic-gradient(var(--ok) 0deg ${a}deg, var(--bad) ${a}deg 360deg)`;
    donut.dataset.label = share.toFixed(1) + "% attack";
  }

  const fams = Object.entries(s.family_counts).sort((a, b) => b[1] - a[1]);
  if (!fams.length) {
    $("fam-bars").innerHTML = '<p class="empty">No attacks found yet. Upload a file or check a flow.</p>';
  } else {
    const max = fams[0][1];
    $("fam-bars").innerHTML = fams.map(([k, v]) =>
      `<div class="r"><span>${k}</span><div class="track"><div class="fill" style="width:${(v / max) * 100}%"></div></div><span>${fmt(v)}</span></div>`).join("");
  }
}

async function refreshStats() { renderStats(await api("/api/stats")); }

/* ---------- upload ---------- */
$("upload").onclick = async () => {
  const f = $("file").files[0], msg = $("upload-msg");
  msg.className = "msg";
  if (!f) { msg.className = "msg err"; msg.textContent = "Choose a CSV file first."; return; }
  const fd = new FormData(); fd.append("file", f);
  $("upload").disabled = true; msg.textContent = "Analyzing...";
  try {
    const r = await api("/api/upload", { method: "POST", body: fd });
    const s = r.summary;
    let t = `Analyzed ${fmt(s.total)} rows: ${fmt(s.normal)} normal, ${fmt(s.attack)} attack.`;
    if (s.dropped_rows) t += ` ${s.dropped_rows} unusable rows were skipped.`;
    msg.textContent = t;
    $("rows").innerHTML = "<tr><th>Row</th><th>Result</th><th>Attack type</th></tr>" +
      r.rows.map((x) => `<tr><td>${x.row}</td><td class="${x.prediction[0]}">${x.prediction}</td><td>${x.attack_family}</td></tr>`).join("");
    $("rows-panel").hidden = false;
    refreshStats();
  } catch (e) { msg.className = "msg err"; msg.textContent = e.message; }
  $("upload").disabled = false;
};

/* ---------- single flow ---------- */
let FEATURES = [];

async function buildFields() {
  FEATURES = (await api("/api/features")).features;
  $("fields").innerHTML = FEATURES.map((f) =>
    `<div><label for="f-${f}">${f}</label><input id="f-${f}" type="number" step="any" value="0"></div>`).join("");
}

async function loadSample(type) {
  const r = await api("/api/sample?type=" + type);
  FEATURES.forEach((f) => { $("f-" + f).value = r.features[f]; });
  const el = $("result");
  el.className = "result info";
  el.textContent = "Sample loaded. Click Check flow.";
}

$("s-attack").onclick = () => loadSample("attack");
$("s-normal").onclick = () => loadSample("normal");

$("predict").onclick = async () => {
  const features = {};
  FEATURES.forEach((f) => { features[f] = parseFloat($("f-" + f).value) || 0; });
  const el = $("result");
  try {
    const r = await api("/api/predict", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ features }),
    });
    const attack = r.prediction === "ATTACK";
    el.className = "result " + (attack ? "bad" : "ok");
    el.innerHTML = `<b>${attack ? "ATTACK / SUSPICIOUS" : "NORMAL"}</b>` +
      (attack ? `<br>Likely attack type: ${r.attack_family}` : "<br>This flow looks like regular traffic.");
    refreshStats();
  } catch (e) { el.className = "result bad"; el.textContent = e.message; }
};

$("reset").onclick = async () => {
  renderStats(await api("/api/reset", { method: "POST" }));
  $("rows-panel").hidden = true;
  $("upload-msg").textContent = "";
};

Promise.all([refreshStats(), buildFields()]).catch((e) => {
  $("upload-msg").className = "msg err";
  $("upload-msg").textContent = "Could not reach the server: " + e.message;
});