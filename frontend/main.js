async function uploadFile() {
    const fileInput = document.getElementById("fileInput");
    const statusMessage = document.getElementById("statusMessage");

    if (fileInput.files.length === 0) {
        statusMessage.textContent = "❌ Please select a file.";
        return;
    }

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append("file", file);

    statusMessage.textContent = "⏳ Uploading...";

    try {
        const response = await fetch("http://localhost:8000/upload-mesh/", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            console.error("Server Error:", errorData);
            statusMessage.textContent = `❌ Upload failed: ${errorData.detail}`;
            return;
        }

        const data = await response.json();
        console.log("✅ Upload Success:", data);

        if (data.mesh_url) {
            statusMessage.textContent = "✅ File processed successfully!";
            load3DModel(data.mesh_url);
        } else {
            statusMessage.textContent = "❌ Error processing file. No mesh URL received.";
        }
    } catch (error) {
        console.error("❌ Upload failed:", error);
        statusMessage.textContent = "❌ Upload failed. Check console for details.";
    }
}


async function sendMessage() {
    const chatInput = document.getElementById("chat-input");
    const chatBox = document.getElementById("chat-box");
    const userMessage = chatInput.value.trim();
    if (!userMessage) return;

    // Append user message (here we assume it’s plain text)
    chatBox.innerHTML += `<div class="chat-message user-message">${userMessage}</div>`;
    chatInput.value = "";

    try {
        const response = await fetch("http://localhost:8000/chat/", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ user_input: userMessage }),
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const data = await response.json();
        console.info(data)
        // Assume data.response contains HTML-formatted rich text from the AI
        chatBox.innerHTML += data.response;

        // Auto-scroll to the latest message
        chatBox.scrollTop = chatBox.scrollHeight;
    } catch (error) {
        console.error("❌ Chatbot error:", error);
        chatBox.innerHTML += `<div class="chat-message bot-message"><b>Error:</b> Failed to fetch response.</div>`;
    }
}

