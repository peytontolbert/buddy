
// Function to handle login form submission
document.getElementById('loginForm').addEventListener('submit', function(event) {
    event.preventDefault();
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;
    console.log("Username:", username, "Password:", password);
});

// Function to fetch world state from the backend
function fetchWorldState() {
    const worldState = {
        room: { width: 10, height: 10 },
        character: { x: 0, y: 0, status: 'idle' }
    };
    const canvas = document.getElementById('gameCanvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.beginPath();
    ctx.arc(worldState.character.x * 50, worldState.character.y * 50, 10, 0, Math.PI * 2);
    ctx.fillStyle = "blue";
    ctx.fill();
    ctx.closePath();
}

// Function to send a chat message
document.getElementById('sendButton').addEventListener('click', function() {
    const message = document.getElementById('chatInput').value;
    const chatArea = document.getElementById('chatArea');
    chatArea.innerHTML += `<p>You: ${message}</p>`;
    document.getElementById('chatInput').value = '';
});

// Fetch the world state periodically
setInterval(fetchWorldState, 5000);
