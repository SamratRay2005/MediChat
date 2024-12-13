document.getElementById('chat-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    const input = document.getElementById('user-input');
    const userInput = input.value.trim();
    if (!userInput) return;

    // Add user message to chat
    addMessage(userInput, false);
    input.value = '';

    // Send the message to Flask backend
    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: userInput })
        });

        const result = await response.json();

        // Format bot message with bolded diseases and recommendations on a new line
        const formattedResponse = formatBotResponse(result.response);
        addMessage(formattedResponse, true);
    } catch (error) {
        console.error("Error:", error);
        addMessage("Something went wrong. Please try again later.", true);
    }
});

function addMessage(text, isBot) {
    const chatContainer = document.getElementById('chat-container');
    const botIconURL = chatContainer.getAttribute('data-bot-icon');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isBot ? 'bot' : 'user'}`;

    if (isBot) {
        // Added bot icon container along with message text for consistency
        messageDiv.innerHTML = `
            <div class="bot-info">
                <img src="${botIconURL}" alt="Bot Icon" class="bot-icon"/>
                <div class="message-text">${text}</div>
            </div>
        `;
    } else {
        messageDiv.innerHTML = `<p class="message-text">${text}</p>`;
    }

    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function formatBotResponse(response) {
    // Format the response into paragraphs, bullet points, and emphasized text
    return response
        // Bold disease names
        .replace(/(Based on your symptoms, possible conditions are:)/, "<strong>$1</strong>")
        // Ensure newline before "Additional guidance"
        .replace(/(Additional guidance:)/, "<br><br><strong>$1</strong>")
        // Format emphasis for emergency guidance
        .replace(/(\*\*(.*?)\*\*)/g, "<strong>$2</strong>")
        // Format bullet points
        .replace(/\*(.*?)\*/g, "<li>$1</li>")
        // Wrap lists inside <ul> tags
        .replace(/(<li>.*?<\/li>)/g, "<ul>$1</ul>")
        // Paragraph breaks
        .replace(/\n/g, "<br>");
}
