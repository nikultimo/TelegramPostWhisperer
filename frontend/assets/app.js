const DEFAULT_MESSAGE =
  "Привет! Хочу посоветовать тебе пару классных сервисов для общения и подарков.\n\n" +
  "1. [Together](https://together.sevostianovs.ru/) — помогает парам планировать общие активности и укреплять отношения. " +
  "Загляни и подключи @RelationshipTogetherBot, чтобы напоминания и идеи были всегда под рукой.\n" +
  "2. [WishShare](https://wishshare.sevostianovs.ru/) — твой персональный wishlist с ботом @happywishlistbot. " +
  "Делись желаниями, собирай подарки и удивляй близких.\n\n" +
  "Переходи по ссылкам, попробуй и расскажи, как тебе 👍";

const form = document.getElementById("broadcast-form");
const messageInput = document.getElementById("message");
const previewText = document.getElementById("preview-text");
const previewPhotos = document.getElementById("preview-photos");
const previewVideos = document.getElementById("preview-videos");
const previewButtons = document.getElementById("preview-buttons");
const parseModeSelect = document.getElementById("parse-mode");
const parseModeInput = document.getElementById("parse-mode-input");
const inlineHidden = document.getElementById("inline-keyboard");
const builder = document.getElementById("inline-builder");
const addRowButton = document.getElementById("add-row");
const resetKeyboardButton = document.getElementById("reset-keyboard");
const buttonTemplate = document.getElementById("button-template");
const rowTemplate = document.getElementById("inline-row-template");
const csvSummary = document.getElementById("csv-summary");
const photosInput = document.getElementById("photos");
const videosInput = document.getElementById("videos");
const resultBox = document.getElementById("result");
const disablePreviewCheckbox = document.getElementById("disable-preview");
const attachCaptionCheckbox = document.getElementById("attach-caption");
const enhanceButton = document.getElementById("enhance-button");

const photoObjectUrls = new Set();
const videoObjectUrls = new Set();
let currentPhotos = [];
let currentVideos = [];
let availableBots = [];
let availableCsvFiles = [];
const botSelectionContainer = document.getElementById("bot-selection-container");
const botSelectionHelper = document.getElementById("bot-selection-helper");
const botSelectionFieldset = document.getElementById("bot-selection-fieldset");
const tokenFieldset = document.getElementById("token-fieldset");
const tokenInput = document.getElementById("token");
const csvInput = document.getElementById("csv");
const csvModeUpload = document.getElementById("csv-mode-upload");
const csvModeSelect = document.getElementById("csv-mode-select");
const csvUploadContainer = document.getElementById("csv-upload-container");
const csvSelectContainer = document.getElementById("csv-select-container");
let csvFileSelect = document.getElementById("csv-file-select");
const csvSelectSummary = document.getElementById("csv-select-summary");

marked.setOptions({
  breaks: true,
  gfm: true,
});

messageInput.value = DEFAULT_MESSAGE;
renderPreview();
updateInlineKeyboard();
loadBots();
loadCsvFiles();

// CSV mode selection
if (csvModeUpload && csvModeSelect) {
  // Set initial state
  if (csvModeUpload.checked) {
    csvInput.setAttribute("required", "required");
    csvFileSelect.removeAttribute("required");
  }
  
  csvModeUpload.addEventListener("change", () => {
    if (csvModeUpload.checked) {
      csvUploadContainer.classList.remove("hidden");
      csvSelectContainer.classList.add("hidden");
      csvInput.setAttribute("required", "required");
      csvFileSelect.removeAttribute("required");
    }
  });
  
  csvModeSelect.addEventListener("change", () => {
    if (csvModeSelect.checked) {
      csvUploadContainer.classList.add("hidden");
      csvSelectContainer.classList.remove("hidden");
      csvInput.removeAttribute("required");
      csvFileSelect.setAttribute("required", "required");
    }
  });
}

messageInput.addEventListener("input", renderPreview);
parseModeSelect.addEventListener("change", () => {
  parseModeInput.value = parseModeSelect.value;
  renderPreview();
});

if (enhanceButton) {
  console.log("Enhance button found and initialized");
  enhanceButton.addEventListener("click", async (event) => {
    event.preventDefault();
    event.stopPropagation();
  const currentText = messageInput.value.trim();
  
  if (!currentText) {
    showResult("Введите текст сообщения для улучшения", "error");
    return;
  }

  enhanceButton.disabled = true;
  enhanceButton.textContent = "⏳ Улучшаю...";
  showResult("Улучшаю текст...", "info");

  try {
    console.log("Sending enhance request...", {
      messageLength: currentText.length,
      parseMode: parseModeSelect.value
    });
    
    const response = await fetch("/api/enhance", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message: currentText,
        parse_mode: parseModeSelect.value,
      }),
    });

    console.log("Response status:", response.status, response.statusText);

    let data;
    let responseText = "";
    try {
      responseText = await response.text();
      console.log("Response text:", responseText.substring(0, 500));
      data = JSON.parse(responseText);
    } catch (jsonError) {
      console.error("JSON parse error:", jsonError);
      throw new Error(`Invalid JSON response: ${responseText?.substring(0, 200) || 'empty response'}`);
    }

    if (response.ok) {
      if (data.enhanced_message) {
        console.log("Enhanced message received, length:", data.enhanced_message.length);
        messageInput.value = data.enhanced_message;
        renderPreview();
        showResult("Текст успешно улучшен! ✨", "success");
      } else {
        console.error("No enhanced_message in response:", data);
        showResult("Ошибка: ответ не содержит улучшенного текста", "error");
      }
    } else {
      console.error("API error:", data);
      showResult(
        `Ошибка ${response.status}: ${data.detail ?? data.message ?? response.statusText}`,
        "error"
      );
    }
  } catch (error) {
    console.error("Enhance error:", error);
    showResult(`Не удалось улучшить текст: ${error.message || error}`, "error");
  } finally {
    enhanceButton.disabled = false;
    enhanceButton.textContent = "✨ Enhance";
  }
  });
} else {
  console.error("Enhance button not found in DOM");
}

csvInput.addEventListener("change", handleCsvChange);
photosInput.addEventListener("change", handlePhotoChange);
videosInput.addEventListener("change", handleVideoChange);

addRowButton.addEventListener("click", () => {
  const row = createRow();
  builder.appendChild(row);
  addButtonToRow(row);
  updateInlineKeyboard();
});

resetKeyboardButton.addEventListener("click", () => {
  builder.innerHTML = "";
  updateInlineKeyboard();
});

document.addEventListener("paste", handlePaste);

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  updateInlineKeyboard();

  // Check CSV file selection
  const isUploadMode = csvModeUpload && csvModeUpload.checked;
  const isSelectMode = csvModeSelect && csvModeSelect.checked;
  
  if (isUploadMode && !csvInput.files.length) {
    showResult("Добавьте CSV файл с пользователями или выберите из папки data.", "error");
    return;
  }
  
  if (isSelectMode && !csvFileSelect.value) {
    showResult("Выберите CSV файл из папки data или загрузите файл.", "error");
    return;
  }

  // Get selected bot tokens
  const selectedBotTokens = getSelectedBotTokens();
  if (availableBots.length > 0 && selectedBotTokens.length === 0) {
    showResult("Выберите хотя бы одного бота для отправки.", "error");
    return;
  }

  // If no bots.yaml exists, require manual token
  if (availableBots.length === 0 && !tokenInput.value.trim()) {
    showResult("Введите токен бота или создайте bots.yaml файл.", "error");
    return;
  }

  showResult("Отправляем рассылку...", "info");

  const formData = new FormData(form);

  // Add selected bots if available
  if (selectedBotTokens.length > 0) {
    formData.set("selected_bots", JSON.stringify(selectedBotTokens));
    formData.delete("token"); // Remove token if using selected bots
  } else if (tokenInput.value.trim()) {
    formData.set("token", tokenInput.value.trim());
  }

  if (parseModeSelect.value === "None") {
    formData.set("parse_mode", "None");
  } else {
    formData.set("parse_mode", parseModeSelect.value);
  }

  if (!disablePreviewCheckbox.checked) {
    formData.delete("disable_preview");
  }

  if (!attachCaptionCheckbox.checked) {
    formData.delete("attach_message_to_first_photo");
  }

  // Handle CSV file: either uploaded or selected from data directory
  if (isUploadMode && csvInput.files.length > 0) {
    const csvFile = csvInput.files[0];
    const refreshed = await createFreshFile(csvFile);
    formData.set("csv_file", refreshed, refreshed.name);
  } else if (isSelectMode && csvFileSelect.value) {
    formData.set("csv_file_name", csvFileSelect.value);
    formData.delete("csv_file");
  }

  formData.delete("photos");
  currentPhotos.forEach((file) => {
    formData.append("photos", file, file.name);
  });

  formData.delete("videos");
  currentVideos.forEach((file) => {
    formData.append("videos", file, file.name);
  });

  try {
    const response = await fetch("/api/send", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();

    if (response.ok || response.status === 207) {
      const failedCount = data.failed?.length || 0;
      const successCount = data.delivered || 0;
      const totalCount = data.total || 0;
      
      let resultMessage = `Готово! Отправлено: ${successCount} из ${totalCount}`;
      if (data.bots && data.bots.length > 1) {
        resultMessage += `\nБотов использовано: ${data.bots.length}`;
      }
      
      if (failedCount > 0) {
        showResult(
          `Отправлено: ${successCount} из ${totalCount}\nНе доставлено: ${failedCount}`,
          "partial"
        );
      } else {
        showResult(resultMessage, "success");
      }
    } else {
      showResult(
        `Ошибка ${response.status}: ${data.detail ?? response.statusText}`,
        "error",
      );
    }
  } catch (error) {
    showResult(`Не удалось выполнить запрос: ${error}`, "error");
  }
});

function createRow() {
  const fragment = rowTemplate.content.cloneNode(true);
  const row = fragment.querySelector(".inline-row");
  const addButton = row.querySelector(".add-button");
  addButton.addEventListener("click", () => {
    addButtonToRow(row);
    updateInlineKeyboard();
  });
  return row;
}

function addButtonToRow(row) {
  const fragment = buttonTemplate.content.cloneNode(true);
  const entry = fragment.querySelector(".button-entry");
  const inputs = entry.querySelectorAll("input");
  const removeButton = entry.querySelector(".remove-button");

  inputs.forEach((input) => {
    input.addEventListener("input", () => {
      updateInlineKeyboard();
    });
  });

  removeButton.addEventListener("click", () => {
    entry.remove();
    if (!row.querySelector(".button-entry")) {
      row.remove();
    }
    updateInlineKeyboard();
  });

  row.querySelector(".row-buttons").appendChild(entry);
}

function updateInlineKeyboard() {
  const rows = [];
  builder.querySelectorAll(".inline-row").forEach((row) => {
    const buttons = [];
    row.querySelectorAll(".button-entry").forEach((entry) => {
      const text = entry.querySelector(".btn-text").value.trim();
      const url = entry.querySelector(".btn-url").value.trim();
      const callback = entry.querySelector(".btn-callback").value.trim();

      if (!text) {
        return;
      }

      const button = { text };
      if (url) {
        button.url = url;
      } else if (callback) {
        button.callback_data = callback;
      }
      buttons.push(button);
    });

    if (buttons.length) {
      rows.push(buttons);
    }
  });

  const normalizedRows = rows.flatMap((buttons) =>
    buttons.map((button) => [button]),
  );

  inlineHidden.value = normalizedRows.length
    ? JSON.stringify(normalizedRows)
    : "";
  renderButtons(normalizedRows);
}

function renderPreview() {
  const mode = parseModeSelect.value;
  const text = messageInput.value;

  if (mode === "HTML") {
    previewText.innerHTML = text;
  } else if (mode === "None") {
    previewText.textContent = text;
  } else {
    previewText.innerHTML = marked.parse(text);
  }
}

function renderButtons(rows) {
  previewButtons.innerHTML = "";

  const normalizedRows = rows
    .map((buttons) => buttons.filter((btn) => btn.text && btn.text.trim()))
    .filter((buttons) => buttons.length);

  normalizedRows.forEach((buttons) => {
    buttons.forEach((button) => {
      const rowContainer = document.createElement("div");
      rowContainer.className = "button-row column";

      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "inline-button";
      btn.textContent = button.text;

      rowContainer.appendChild(btn);
      previewButtons.appendChild(rowContainer);
    });
  });
}

function handleCsvChange() {
  csvSummary.textContent = "";
  const [file] = csvInput.files;
  if (!file) {
    csvSummary.textContent = "";
    return;
  }

  const reader = new FileReader();
  reader.onload = (event) => {
    const text = event.target.result;
    const rows = text
      .split(/\r?\n/)
      .map((row) => row.trim())
      .filter(Boolean);
    const count = Math.max(rows.length - 1, 0);
    csvSummary.textContent = `Строк: ~${count || rows.length}`;
  };
  reader.readAsText(file);
}

function handlePhotoChange() {
  currentPhotos = Array.from(photosInput.files);
  renderPhotoPreview();
}

function handleVideoChange() {
  currentVideos = Array.from(videosInput.files);
  renderVideoPreview();
}

async function createFreshFile(file) {
  const buffer = await file.arrayBuffer();
  return new File([buffer], file.name, { type: file.type || "text/csv" });
}

function renderPhotoPreview() {
  photoObjectUrls.forEach((url) => URL.revokeObjectURL(url));
  photoObjectUrls.clear();
  previewPhotos.innerHTML = "";

  const dataTransfer =
    typeof DataTransfer !== "undefined" ? new DataTransfer() : null;
  currentPhotos.forEach((file, index) => {
    if (dataTransfer) {
      dataTransfer.items.add(file);
    }
    const url = URL.createObjectURL(file);
    photoObjectUrls.add(url);

    const tile = document.createElement("div");
    tile.className = "photo-tile";

    const img = document.createElement("img");
    img.src = url;
    img.alt = file.name;

    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.className = "remove-photo";
    removeBtn.textContent = "×";
    removeBtn.dataset.index = String(index);
    removeBtn.dataset.type = "photo";
    removeBtn.addEventListener("click", onRemoveMediaClick);

    tile.append(img, removeBtn);
    previewPhotos.appendChild(tile);
  });

  if (dataTransfer) {
    photosInput.files = dataTransfer.files;
  }
  if (currentPhotos.length === 0) {
    photosInput.value = "";
  }
}

function renderVideoPreview() {
  videoObjectUrls.forEach((url) => URL.revokeObjectURL(url));
  videoObjectUrls.clear();
  previewVideos.innerHTML = "";

  const dataTransfer =
    typeof DataTransfer !== "undefined" ? new DataTransfer() : null;
  currentVideos.forEach((file, index) => {
    if (dataTransfer) {
      dataTransfer.items.add(file);
    }
    const url = URL.createObjectURL(file);
    videoObjectUrls.add(url);

    const tile = document.createElement("div");
    tile.className = "photo-tile";

    const video = document.createElement("video");
    video.src = url;
    video.controls = true;
    video.style.maxWidth = "100%";
    video.style.maxHeight = "200px";

    const nameLabel = document.createElement("div");
    nameLabel.textContent = file.name;
    nameLabel.style.fontSize = "0.8rem";
    nameLabel.style.marginTop = "0.5rem";
    nameLabel.style.color = "#999";

    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.className = "remove-photo";
    removeBtn.textContent = "×";
    removeBtn.dataset.index = String(index);
    removeBtn.dataset.type = "video";
    removeBtn.addEventListener("click", onRemoveMediaClick);

    tile.append(video, nameLabel, removeBtn);
    previewVideos.appendChild(tile);
  });

  if (dataTransfer) {
    videosInput.files = dataTransfer.files;
  }
  if (currentVideos.length === 0) {
    videosInput.value = "";
  }
}

function onRemoveMediaClick(event) {
  const index = Number(event.currentTarget.dataset.index);
  const type = event.currentTarget.dataset.type;
  if (Number.isInteger(index)) {
    if (type === "video") {
      removeVideo(index);
    } else {
      removePhoto(index);
    }
  }
}

function removePhoto(index) {
  currentPhotos.splice(index, 1);
  renderPhotoPreview();
}

function removeVideo(index) {
  currentVideos.splice(index, 1);
  renderVideoPreview();
}

function handlePaste(event) {
  const items = event.clipboardData?.items;
  if (!items) {
    return;
  }

  let added = false;
  Array.from(items).forEach((item) => {
    if (item.kind !== "file") {
      return;
    }
    const file = item.getAsFile();
    if (!file || !file.type.startsWith("image/")) {
      return;
    }

    const extension = file.type.split("/")[1] || "png";
    const name =
      file.name && file.name.trim()
        ? file.name
        : `pasted_image_${Date.now()}_${Math.random()
            .toString(16)
            .slice(2)}.${extension}`;
    const normalized = new File([file], name, { type: file.type });
    currentPhotos.push(normalized);
    added = true;
  });

  if (added) {
    renderPhotoPreview();
    event.preventDefault();
  }
}

function showResult(message, type) {
  resultBox.classList.remove("hidden", "success", "error", "partial");
  resultBox.textContent = message;
  if (type === "success") {
    resultBox.classList.add("success");
    resultBox.classList.remove("error");
  } else if (type === "partial") {
    resultBox.classList.add("partial");
  } else if (type === "error") {
    resultBox.classList.add("error");
    resultBox.classList.remove("success");
  } else {
    resultBox.classList.remove("success", "error", "partial");
  }
}

async function loadBots() {
  if (!botSelectionFieldset || !tokenFieldset || !botSelectionHelper) {
    console.error("Bot selection elements not found in DOM");
    return;
  }
  
  try {
    const response = await fetch("/api/bots");
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    const data = await response.json();
    
    if (data.exists && data.bots && data.bots.length > 0) {
      availableBots = data.bots;
      renderBotSelection();
      botSelectionFieldset.classList.remove("hidden");
      tokenFieldset.classList.add("hidden");
      if (tokenInput) tokenInput.removeAttribute("required");
      if (botSelectionHelper) {
        botSelectionHelper.textContent = `Доступно ботов: ${data.bots.length}. Выберите одного или нескольких.`;
      }
    } else {
      availableBots = [];
      botSelectionFieldset.classList.add("hidden");
      tokenFieldset.classList.remove("hidden");
      if (tokenInput) tokenInput.setAttribute("required", "required");
      if (botSelectionHelper) {
        botSelectionHelper.textContent = "Файл bots.yaml не найден. Введите токен вручную.";
      }
    }
  } catch (error) {
    console.error("Failed to load bots:", error);
    availableBots = [];
    if (botSelectionFieldset) botSelectionFieldset.classList.add("hidden");
    if (tokenFieldset) tokenFieldset.classList.remove("hidden");
    if (tokenInput) tokenInput.setAttribute("required", "required");
    if (botSelectionHelper) {
      botSelectionHelper.textContent = "Ошибка загрузки списка ботов. Используйте поле токена.";
    }
  }
}

function renderBotSelection() {
  if (!botSelectionContainer) {
    console.error("botSelectionContainer not found");
    return;
  }
  
  botSelectionContainer.innerHTML = "";
  
  if (availableBots.length === 0) {
    if (botSelectionHelper) {
      botSelectionHelper.textContent = "Боты не найдены в bots.yaml";
    }
    return;
  }
  
  availableBots.forEach((bot, index) => {
    const label = document.createElement("label");
    label.className = "toggle";
    label.style.display = "flex";
    label.style.alignItems = "center";
    label.style.marginBottom = "0.5rem";
    
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.value = bot.token;
    checkbox.id = `bot-${index}`;
    checkbox.dataset.botName = bot.name;
    
    const span = document.createElement("span");
    span.textContent = bot.name;
    span.style.marginLeft = "0.5rem";
    
    label.appendChild(checkbox);
    label.appendChild(span);
    botSelectionContainer.appendChild(label);
  });
  
  if (botSelectionHelper) {
    botSelectionHelper.textContent = `Доступно ботов: ${availableBots.length}. Выберите одного или нескольких.`;
  }
}

function getSelectedBotTokens() {
  const checkboxes = botSelectionContainer.querySelectorAll('input[type="checkbox"]:checked');
  return Array.from(checkboxes).map(cb => cb.value);
}

async function loadCsvFiles() {
  try {
    const response = await fetch("/api/csv-files");
    const data = await response.json();
    
    if (data.exists && data.files && data.files.length > 0) {
      availableCsvFiles = data.files;
      renderCsvFileSelection();
    } else {
      availableCsvFiles = [];
      if (csvFileSelect) {
        csvFileSelect.innerHTML = '<option value="">CSV файлы не найдены в папке data</option>';
      }
    }
  } catch (error) {
    console.error("Failed to load CSV files:", error);
    availableCsvFiles = [];
    if (csvFileSelect) {
      csvFileSelect.innerHTML = '<option value="">Ошибка загрузки списка файлов</option>';
    }
  }
}

function renderCsvFileSelection() {
  const selectElement = document.getElementById("csv-file-select");
  if (!selectElement) return;
  
  selectElement.innerHTML = '<option value="">Выберите файл...</option>';
  
  availableCsvFiles.forEach((file) => {
    const option = document.createElement("option");
    option.value = file.name;
    const userCount = file.user_count !== undefined ? file.user_count : null;
    if (userCount !== null && userCount > 0) {
      option.textContent = `${file.name} (${userCount} пользователей)`;
    } else {
      option.textContent = file.name;
    }
    selectElement.appendChild(option);
  });
  
  // Добавляем обработчик изменения для обновления summary
  selectElement.removeEventListener("change", updateCsvFileSummary);
  selectElement.addEventListener("change", updateCsvFileSummary);
}

function updateCsvFileSummary() {
  const selectElement = document.getElementById("csv-file-select");
  if (!selectElement) return;
  
  const selectedFileName = selectElement.value;
  const summaryElement = csvSelectSummary || csvSummary;
  
  if (!summaryElement) return;
  
  if (!selectedFileName) {
    summaryElement.textContent = "";
    return;
  }
  
  const selectedFile = availableCsvFiles.find(f => f.name === selectedFileName);
  if (selectedFile && selectedFile.user_count !== undefined && selectedFile.user_count > 0) {
    summaryElement.textContent = `Пользователей: ${selectedFile.user_count}`;
  } else {
    summaryElement.textContent = "";
  }
}

