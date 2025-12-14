const API_URL = "http://127.0.0.1:8000/predict";

const itemsDiv = document.getElementById("items");
const overallDiv = document.getElementById("overall");

let items = [];

// utils
function formatFoodName(name) {
  if (!name) return "";
  return name.replace(/[_-]+/g, " ").trim();
}

function goHome() {
  window.location.href = "index.html";
}


function newItem() {
  return {
    id: crypto.randomUUID(),
    file: null,
    preview: null,
    grams: 180,
    result: null
  };
}

function addNewItem() {
  items.push(newItem());
  render();
}

function removeItem(id) {
  items = items.filter(x => x.id !== id);
  render();
}

function setGrams(id, grams) {
  grams = Number(grams);
  items = items.map(x => x.id === id ? { ...x, grams } : x);
  render();
}

function setPreset(id, grams) {
  setGrams(id, grams);
}

function onFileChange(id, ev) {
  const file = ev.target.files?.[0];
  if (!file) return;

  items = items.map(x => {
    if (x.id !== id) return x;
    return {
      ...x,
      file,
      preview: URL.createObjectURL(file),
      result: null
    };
  });

  render();
}

async function analyze(id) {
  const item = items.find(x => x.id === id);
  if (!item?.file) {
    alert("Please upload an image first.");
    return;
  }

  const fd = new FormData();
  fd.append("file", item.file);
  fd.append("grams", String(item.grams));

  try {
    const res = await fetch(API_URL, {
      method: "POST",
      body: fd
    });

    const data = await res.json();

    items = items.map(x =>
      x.id === id ? { ...x, result: data } : x
    );

    render();

  } catch (e) {
    alert("API connection failed. Is backend running on 127.0.0.1:8000 ?");
  }
}

function computeOverall() {
  const total = { calories: 0, protein: 0, fat: 0, carbs: 0 };

  for (const it of items) {
    const r = it.result;
    if (r && r.food_detected) {
      total.calories += Number(r.calories || 0);
      total.protein += Number(r.protein || 0);
      total.fat += Number(r.fat || 0);
      total.carbs += Number(r.carbs || 0);
    }
  }

  return total;
}

function render() {
  itemsDiv.innerHTML = "";

  for (const item of items) {
    const presets = [
      { label: "50g", grams: 50 },
      { label: "100g", grams: 100 },
      { label: "150g", grams: 150 },
      { label: "200g", grams: 200 },
      { label: "300g", grams: 300 },
      { label: "1 portion", grams: 180 }
    ];

    const chipsHtml = presets.map(p => {
      const active = item.grams === p.grams ? "active" : "";
      return `<button class="chip ${active}"
        onclick="setPreset('${item.id}', ${p.grams})">${p.label}</button>`;
    }).join("");

    let resultHtml = "";
    const r = item.result;

    if (r) {
      if (!r.food_detected) {
        resultHtml = `
          <div class="result">
            <div class="food-title">Not a food image</div>
            <p class="conf">
              Please upload a food photo
              (confidence: ${(r.confidence * 100).toFixed(1)}%)
            </p>
          </div>
        `;
      } else {
        resultHtml = `
          <div class="result">
            <div class="food-title">
              ${formatFoodName(r.food_name)}
            </div>
            <p class="conf">
              Confidence: ${(r.confidence * 100).toFixed(2)}%
              • Portion: ${item.grams} g
            </p>
            <div class="grid">
              <div class="kv">🔥 Calories: <b>${r.calories}</b> kcal</div>
              <div class="kv">💪 Protein: <b>${r.protein}</b> g</div>
              <div class="kv">🧈 Fat: <b>${r.fat}</b> g</div>
              <div class="kv">🍞 Carbs: <b>${r.carbs}</b> g</div>
            </div>
          </div>
        `;
      }
    }

    const el = document.createElement("section");
    el.className = "item";

    el.innerHTML = `
      <div class="preview">
        ${item.preview
          ? `<img src="${item.preview}" alt="preview" />`
          : `<div class="upload-area">
              <div style="font-weight:900;color:#dbe6ff;">Upload Image</div>
              <div style="font-size:12px;">JPG/PNG • clear food photo</div>
            </div>`
        }
      </div>

      <div class="right">
        <div class="chips">${chipsHtml}</div>

        <div class="slider-row">
          <input class="slider" type="range"
            min="10" max="2000" step="10"
            value="${item.grams}"
            oninput="setGrams('${item.id}', this.value)" />
          <div class="gram-badge">${item.grams} g</div>
        </div>

        <div class="actions">
          <label class="file">
            <input type="file" accept="image/*" hidden
              onchange="onFileChange('${item.id}', event)" />
            📷 Choose Image
          </label>

          <button class="btn-analyze" onclick="analyze('${item.id}')">Analyze</button>
          <button class="btn-add" onclick="addNewItem()">Add New</button>
          <button class="btn-remove" onclick="removeItem('${item.id}')">Remove</button>
        </div>

        ${resultHtml}
      </div>
    `;

    itemsDiv.appendChild(el);
  }

  const total = computeOverall();
  overallDiv.textContent =
    `🔥 Calories: ${total.calories.toFixed(1)} kcal | ` +
    `💪 Protein: ${total.protein.toFixed(1)} g | ` +
    `🧈 Fat: ${total.fat.toFixed(1)} g | ` +
    `🍞 Carbs: ${total.carbs.toFixed(1)} g`;
}

// init
items.push(newItem());
render();
