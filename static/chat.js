// ============================================================
// DataCompany RAG Chatbot
// Features: voice input/output, typing animation, dark mode,
//           tenant greeting, suggestions, export chat
// ============================================================

const chat    = document.getElementById("chat");
const input   = document.getElementById("input");
const sendBtn = document.getElementById("send-btn");

let tenantName  = "";
let isDark      = false;
let isListening = false;
let recognition = null;
let chatLog     = [];

const NOT_FOUND = "this information is not covered";


// ============================================================
// TENANT NAME GREETING
// ============================================================

function submitName() {
    const val  = document.getElementById("name-input").value.trim();
    tenantName = val || "there";
    document.getElementById("name-screen").style.display = "none";
    document.getElementById("greeting").textContent =
        `Hello ${tenantName}, how can I help you today?`;
    sessionStorage.setItem("tenantName", tenantName);
}

window.addEventListener("load", () => {
    const saved = sessionStorage.getItem("tenantName");
    if (saved) {
        tenantName = saved;
        document.getElementById("name-screen").style.display = "none";
        document.getElementById("greeting").textContent =
            `Hello ${tenantName}, how can I help you today?`;
    }

    if (localStorage.getItem("darkMode") === "true") {
        toggleDark();
    }
});


// ============================================================
// DARK MODE
// ============================================================

function toggleDark() {
    isDark = !isDark;
    document.body.classList.toggle("dark", isDark);
    document.getElementById("dark-btn").textContent =
        isDark ? "Light mode" : "Dark mode";
    localStorage.setItem("darkMode", isDark);
}


// ============================================================
// AUTO RESIZE TEXTAREA
// ============================================================

function autoResize(el) {
    el.style.height = "auto";
    el.style.height = el.scrollHeight + "px";
}

function handleKey(e) {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
}


// ============================================================
// HIDE WELCOME SCREEN
// ============================================================

function hideWelcome() {
    const welcome = document.getElementById("welcome");
    if (welcome) welcome.remove();
}


// ============================================================
// TYPING ANIMATION
// ============================================================

function typeText(element, text, speed = 18) {
    return new Promise(resolve => {
        let i = 0;
        const cursor = document.createElement("span");
        cursor.style.cssText =
            "display:inline-block;width:2px;height:14px;background:#534AB7;" +
            "margin-left:2px;animation:blink 0.7s infinite;vertical-align:middle;";
        element.appendChild(cursor);

        const interval = setInterval(() => {
            if (i < text.length) {
                cursor.insertAdjacentText("beforebegin", text[i]);
                i++;
                chat.scrollTop = chat.scrollHeight;
            } else {
                clearInterval(interval);
                cursor.remove();
                resolve();
            }
        }, speed);
    });
}


// ============================================================
// APPEND MESSAGES
// ============================================================

async function appendMessage(role, text, meta, suggestions) {
    hideWelcome();

    const msg = document.createElement("div");
    msg.className = `message ${role}`;

    const avatar = document.createElement("div");
    avatar.className = `avatar ${role === "tenant" ? "tenant" : "dc"}`;
    avatar.textContent = role === "tenant"
        ? (tenantName ? tenantName[0].toUpperCase() : "T")
        : "DC";

    const bubble = document.createElement("div");
    bubble.className = "bubble";

    // Append to DOM before animating so element is visible
    msg.appendChild(avatar);
    msg.appendChild(bubble);
    chat.appendChild(msg);
    chat.scrollTop = chat.scrollHeight;

    if (role === "system" && text.toLowerCase().includes(NOT_FOUND)) {
        const answerText = document.createElement("div");
        answerText.textContent = text;

        const notFoundBox = document.createElement("div");
        notFoundBox.className = "not-found";
        notFoundBox.textContent =
            "This question falls outside current policy documents. " +
            "Please contact your housing officer directly.";

        bubble.appendChild(answerText);
        bubble.appendChild(notFoundBox);

    } else {
        const textNode = document.createElement("div");
        bubble.appendChild(textNode);
        await typeText(textNode, text);

        // Speak the answer aloud after typing finishes
        if (role === "system") speakText(text);
    }

    // Metadata — source, confidence, turn number
    if (role === "system" && meta && meta.source && !text.toLowerCase().includes(NOT_FOUND)) {
        const metaDiv = document.createElement("div");
        metaDiv.className = "meta";

        const sourceItem = document.createElement("a");
        sourceItem.className = "meta-item source-link";
        sourceItem.textContent = "Source: " + meta.source.replace(".pdf", "");
        sourceItem.href = "#";
        sourceItem.onclick = (e) => {
            e.preventDefault();
            openPdfModal(`/policies/${meta.source}`, meta.source.replace(".pdf",""));
        };
        sourceItem.title = "Click to open the full policy document";
        metaDiv.appendChild(sourceItem);

        if (meta.confidence) {
            const badge = document.createElement("span");
            const key   = meta.confidence.replace(/\s+/g, "");
            badge.className = `confidence-badge confidence-${key}`;
            badge.textContent = meta.confidence + " confidence";
            metaDiv.appendChild(badge);
        }

        if (meta.turns) {
            const turnItem = document.createElement("span");
            turnItem.className = "meta-item";
            turnItem.textContent = "Turn " + meta.turns;
            metaDiv.appendChild(turnItem);
        }

        bubble.appendChild(metaDiv);

        // Suggested follow-up question chips
        if (suggestions && suggestions.length > 0) {
            const sugDiv = document.createElement("div");
            sugDiv.style.cssText =
                "margin-top:10px;padding-top:10px;border-top:1px solid #E8E7F8;";

            const label = document.createElement("div");
            label.style.cssText = "font-size:11px;color:#888;margin-bottom:6px;";
            label.textContent = "You might also want to ask:";
            sugDiv.appendChild(label);

            const chips = document.createElement("div");
            chips.style.cssText = "display:flex;flex-wrap:wrap;gap:6px;";

            suggestions.forEach(s => {
                const chip = document.createElement("button");
                chip.style.cssText =
                    "font-size:12px;background:#EEEDFE;color:#534AB7;" +
                    "border:1px solid #AFA9EC;padding:4px 10px;" +
                    "border-radius:14px;cursor:pointer;";
                chip.textContent = s;
                chip.onclick = () => sendSuggestion(s);
                chips.appendChild(chip);
            });

            sugDiv.appendChild(chips);
            bubble.appendChild(sugDiv);
        }
    }

    // Add to export log
    chatLog.push({
        role,
        text,
        source: meta ? meta.source : ""
    });

    chat.scrollTop = chat.scrollHeight;
}


// ============================================================
// TYPING INDICATOR
// ============================================================

function showTyping() {
    hideWelcome();
    const typing = document.createElement("div");
    typing.className = "typing";
    typing.id = "typing";
    typing.innerHTML = `
        <div class="avatar dc">DC</div>
        <div class="typing-dots">
            <div class="dot"></div>
            <div class="dot"></div>
            <div class="dot"></div>
        </div>`;
    chat.appendChild(typing);
    chat.scrollTop = chat.scrollHeight;
}

function removeTyping() {
    const t = document.getElementById("typing");
    if (t) t.remove();
}


// ============================================================
// SEND MESSAGE
// ============================================================

async function sendMessage() {
    const question = input.value.trim();
    if (!question) return;

    await appendMessage("tenant", question);
    input.value = "";
    input.style.height = "auto";
    sendBtn.disabled = true;
    showTyping();

    try {
        const res = await fetch("/ask", {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify({ question })
        });

        const data = await res.json();
        removeTyping();

        await appendMessage(
            "system",
            data.answer,
            {
                source:     data.best_source,
                confidence: data.confidence_label,
                turns:      data.history_length
            },
            data.suggestions || []
        );

    } catch (err) {
        removeTyping();
        await appendMessage(
            "system",
            "Connection error. Please check the server and try again.",
            null,
            []
        );
    }

    sendBtn.disabled = false;
    input.focus();
}


// ============================================================
// CLEAR MEMORY
// ============================================================

async function clearMemory() {
    await fetch("/ask", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ question: "clear" })
    });

    const notice = document.createElement("div");
    notice.className = "memory-notice";
    notice.textContent = "Conversation memory cleared";
    chat.appendChild(notice);
    chat.scrollTop = chat.scrollHeight;
    chatLog = [];
}


// ============================================================
// SEND SUGGESTION
// ============================================================

function sendSuggestion(text) {
    input.value = text;
    sendMessage();
}


// ============================================================
// VOICE INPUT
// ============================================================

function toggleVoice() {
    if (!("webkitSpeechRecognition" in window) && !("SpeechRecognition" in window)) {
        alert("Voice input not supported. Please use Chrome.");
        return;
    }
    isListening ? stopListening() : startListening();
}

function startListening() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SR();
    recognition.continuous     = false;
    recognition.interimResults = true;
    recognition.lang           = "en-GB";

    recognition.onstart = () => {
        isListening = true;
        const btn = document.getElementById("voice-btn");
        btn.style.background   = "#FDECEA";
        btn.style.borderColor  = "#E53935";
        btn.style.color        = "#E53935";
    };

    recognition.onresult = (event) => {
        let transcript = "";
        for (let i = event.resultIndex; i < event.results.length; i++) {
            transcript += event.results[i][0].transcript;
        }
        input.value = transcript;
        autoResize(input);
    };

    recognition.onend = () => {
        stopListening();
        if (input.value.trim()) sendMessage();
    };

    recognition.onerror = () => stopListening();
    recognition.start();
}

function stopListening() {
    isListening = false;
    const btn = document.getElementById("voice-btn");
    btn.style.background  = "";
    btn.style.borderColor = "";
    btn.style.color       = "";
    if (recognition) { recognition.stop(); recognition = null; }
}


// ============================================================
// VOICE OUTPUT
// ============================================================

function speakText(text) {
    if (!("speechSynthesis" in window)) return;
    window.speechSynthesis.cancel();

    const clean = text
        .replace(/\*\*/g, "")
        .replace(/\*/g, "")
        .substring(0, 400);

    const utterance  = new SpeechSynthesisUtterance(clean);
    utterance.lang   = "en-GB";
    utterance.rate   = 0.95;
    utterance.pitch  = 1.0;

    const voices  = window.speechSynthesis.getVoices();
    const voice   = voices.find(v => v.lang === "en-GB") ||
                    voices.find(v => v.lang.startsWith("en"));
    if (voice) utterance.voice = voice;

    window.speechSynthesis.speak(utterance);
}


// ============================================================
// EXPORT CHAT
// ============================================================

function exportChat() {
    if (chatLog.length === 0) {
        alert("No conversation to export yet.");
        return;
    }

    const date  = new Date().toLocaleDateString("en-GB");
    let content = "DataCompany Housing Assistant - Chat Export\n";
    content    += `Tenant: ${tenantName || "Unknown"} | Date: ${date}\n`;
    content    += "=".repeat(55) + "\n\n";

    chatLog.forEach(entry => {
        if (entry.role === "tenant") {
            content += `Tenant: ${entry.text}\n\n`;
        } else {
            content += `Assistant: ${entry.text}\n`;
            if (entry.source) {
                content += `Source: ${entry.source.replace(".pdf", "")}\n`;
            }
            content += "\n";
        }
    });

    content += "=".repeat(55) + "\n";
    content += "Generated by DataCompany AI Housing Assistant\n";
    content += "Based on official DataCompany policy documents.\n";

    const blob = new Blob([content], { type: "text/plain" });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href     = url;
    a.download = `DataCompany_Chat_${date.replace(/\//g, "-")}.txt`;
    a.click();
    URL.revokeObjectURL(url);
}

async function openPdfModal(url, title) {
    // Set PDF.js worker
    pdfjsLib.GlobalWorkerOptions.workerSrc =
        "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";

    // Build overlay
    const overlay = document.createElement("div");
    overlay.style.cssText =
        "position:fixed;inset:0;background:rgba(0,0,0,0.75);z-index:200;" +
        "display:flex;align-items:center;justify-content:center;";

    // Build modal
    const modal = document.createElement("div");
    modal.style.cssText =
        "background:#f0f0f0;border-radius:12px;width:85%;height:90vh;" +
        "display:flex;flex-direction:column;overflow:hidden;";

    // Modal header
    const header = document.createElement("div");
    header.style.cssText =
        "padding:12px 18px;background:#2C2060;color:white;display:flex;" +
        "align-items:center;justify-content:space-between;flex-shrink:0;";
    header.innerHTML = `
        <span style="font-size:13px;font-weight:600;">${title}</span>
        <button onclick="this.closest('[style*=fixed]').remove()"
            style="background:rgba(255,255,255,0.15);border:none;color:white;
            border-radius:6px;padding:4px 12px;cursor:pointer;font-size:12px;">
            Close
        </button>`;

    // Scrollable content area
    const content = document.createElement("div");
    content.style.cssText =
        "flex:1;overflow-y:auto;padding:20px;display:flex;" +
        "flex-direction:column;align-items:center;gap:12px;";

    // Loading message
    const loading = document.createElement("div");
    loading.style.cssText = "color:#534AB7;font-size:14px;padding:40px;";
    loading.textContent = "Loading document...";
    content.appendChild(loading);

    modal.appendChild(header);
    modal.appendChild(content);
    overlay.appendChild(modal);
    overlay.onclick = (e) => { if (e.target === overlay) overlay.remove(); };
    document.body.appendChild(overlay);

    try {
        // Extract filename from the URL string
        const filename = url.split('/').pop();

        // Use POST to bypass IDM and other download managers
        const response = await fetch('/api/get-pdf', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename: filename })
        });
        
        if (!response.ok) throw new Error("Failed to fetch PDF");
        const blob = await response.blob();
        const pdfData = await blob.arrayBuffer();

        // Load and render PDF using PDF.js with the buffer
        const loadingTask = pdfjsLib.getDocument({ data: pdfData });
        const pdf = await loadingTask.promise;
        content.removeChild(loading);

        // Render every page as a canvas
        for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
            const page = await pdf.getPage(pageNum);
            const viewport = page.getViewport({ scale: 1.5 });

            const canvas = document.createElement("canvas");
            canvas.width = viewport.width;
            canvas.height = viewport.height;
            canvas.style.cssText = "max-width:100%;box-shadow:0 2px 8px rgba(0,0,0,0.15);border-radius:4px;background:white;";

            await page.render({
                canvasContext: canvas.getContext("2d"),
                viewport
            }).promise;

            content.appendChild(canvas);
        }

    } catch (err) {
        content.innerHTML =
            `<div style="color:#993C1D;padding:20px;font-size:13px;">
                Could not load document: ${err.message}
             </div>`;
    }
}