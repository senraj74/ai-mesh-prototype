import React, { useState } from "react";
import axios from "axios";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, useGLTF } from "@react-three/drei";

function MeshViewer({ meshUrl }) {
    const { scene } = useGLTF(meshUrl);

    return (
        <Canvas style={{ width: "100%", height: "400px" }}>
            <ambientLight intensity={0.5} />
            <directionalLight position={[2, 2, 2]} />
            <OrbitControls />
            <primitive object={scene} />
        </Canvas>
    );
}

export default function ChatBot() {
    const [userInput, setUserInput] = useState("");
    const [messages, setMessages] = useState([]);
    const [selectedFile, setSelectedFile] = useState(null);
    const [uploadMessage, setUploadMessage] = useState("");
    const [meshDownloadUrl, setMeshDownloadUrl] = useState(null);

    const sendMessage = async () => {
        if (!userInput.trim()) return;

        setMessages([...messages, { role: "user", content: userInput }]);
        setUserInput("");

        try {
            const response = await axios.post(
                "http://localhost:8000/chat/",
                { user_input: userInput },
                { headers: { "Content-Type": "application/json" } }
            );

            setMessages([...messages, { role: "bot", content: response.data.response }]);
        } catch (error) {
            console.error("Error fetching chatbot response:", error);
        }
    };

    const handleFileChange = (event) => {
        setSelectedFile(event.target.files[0]);
    };

    const uploadFile = async () => {
        if (!selectedFile) {
            setUploadMessage("❌ Please select a file to upload.");
            return;
        }

        const formData = new FormData();
        formData.append("file", selectedFile);

        try {
            const response = await axios.post(
                "http://localhost:8000/upload-mesh/",
                formData,
                { headers: { "Content-Type": "multipart/form-data" } }
            );

            setUploadMessage(response.data.message);
            setMeshDownloadUrl(response.data.optimized_mesh_url);
        } catch (error) {
            console.error("File upload failed:", error);
            setUploadMessage("❌ File upload failed. Try again.");
        }
    };

    return (
        <div className="chat-container">
            <h1>AI Mesh Chatbot</h1>

            <div className="chat-box">
                {messages.map((msg, i) => (
                    <p key={i} className={msg.role === "user" ? "user-message" : "bot-message"}>
                        {msg.content}
                    </p>
                ))}
            </div>

            <input
                type="text"
                value={userInput}
                onChange={(e) => setUserInput(e.target.value)}
                placeholder="Type a message..."
            />
            <button onClick={sendMessage}>Send</button>

            {/* ✅ File Upload UI */}
            <div className="file-upload">
                <input type="file" onChange={handleFileChange} accept=".obj,.stl,.ply" />
                <button onClick={uploadFile}>Upload 3D Model</button>
                <p>{uploadMessage}</p>
            </div>

            {/* ✅ Render 3D Mesh if available */}
            {meshDownloadUrl && (
                <div>
                    <h3>Optimized Mesh</h3>
                    <MeshViewer meshUrl={meshDownloadUrl} />
                    <a href={meshDownloadUrl} download>
                        <button>Download Optimized Mesh</button>
                    </a>
                </div>
            )}
        </div>
    );
}
