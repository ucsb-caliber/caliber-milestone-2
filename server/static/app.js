const form = document.getElementById("ingestForm");
const out = document.getElementById("out");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  out.textContent = "Uploading + ingesting...";

  const fd = new FormData(form);

  try {
    const res = await fetch("/api/ingest", { method: "POST", body: fd });
    const data = await res.json();
    out.textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    out.textContent = String(err);
  }
});
