import React, { useState } from 'react';
import axios from 'axios';

const BG_COLOR = "#17202A";
const TEXT_COLOR = "#EAECEE";
const BG_GRAY = "#ABB2B9";
const FONT_BOLD = "Helvetica, Arial, sans-serif";
const bot_name = "ChatBot";


const ChatBot = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");

  // Function to send a message to the backend (Python Flask API)
  const getBotResponse = async (msg) => {
    try {
      const response = await axios.post('http://localhost:5000/get_response', {
        message: msg,
      });
      return response.data.response;
    } catch (error) {
      console.error('Error fetching bot response:', error);
      return "Sorry, something went wrong.";
    }
  };

  const handleMessageSubmit = async () => {
    if (!input.trim()) return; // Don't send empty messages

    // Add user message to the chat
    setMessages(prevMessages => [
      ...prevMessages,
      { sender: 'You', message: input }
    ]);

    // Get bot response from the backend and add it to the chat
    const botResponse = await getBotResponse(input);
    setMessages(prevMessages => [
      ...prevMessages,
      { sender: bot_name, message: botResponse }
    ]);

    setInput(""); // Clear the input field
  };

  const handleKeyPress = (event) => {
    if (event.key === 'Enter') {
      handleMessageSubmit();
    }
  };

  return (
    <div style={{ backgroundColor: BG_COLOR, width: '470px', height: '550px', color: TEXT_COLOR, padding: '10px', borderRadius: '10px' }}>
      <div style={{ fontWeight: FONT_BOLD, padding: '10px', color: TEXT_COLOR, textAlign: 'center' }}>
        Welcome
      </div>
      <hr style={{ backgroundColor: BG_GRAY, marginBottom: '10px' }} />

      {/* Message Display Section */}
      <div style={{ height: '75%', overflowY: 'auto', padding: '10px', backgroundColor: BG_COLOR, color: TEXT_COLOR, borderRadius: '5px' }}>
        {messages.map((msg, index) => (
          <div key={index} style={{ paddingBottom: '10px' }}>
            <strong>{msg.sender}:</strong> {msg.message}
          </div>
        ))}
      </div>

      {/* Bottom Input Section */}
      <div style={{ height: '80px', backgroundColor: BG_GRAY, padding: '10px', display: 'flex', alignItems: 'center', borderRadius: '5px' }}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyPress}
          style={{
            width: '80%',
            padding: '10px',
            borderRadius: '5px',
            marginRight: '10px',
            fontSize: '14px',
            backgroundColor: "#2C3E50",
            color: TEXT_COLOR,
            border: 'none'
          }}
        />
        <button
          onClick={handleMessageSubmit}
          style={{
            width: '20%',
            padding: '10px',
            backgroundColor: BG_GRAY,
            color: TEXT_COLOR,
            border: 'none',
            fontWeight: FONT_BOLD,
            cursor: 'pointer',
            borderRadius: '5px'
          }}
        >
          Send
        </button>
      </div>
    </div>
  );
};

export default ChatBot;
