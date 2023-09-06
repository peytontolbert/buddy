// Register form
document.getElementById("register-form").addEventListener("submit", function(e) {
    e.preventDefault();
    let username = document.getElementById("username").value;
    let password = document.getElementById("password").value;
    fetch("/register", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({username, password})
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === "success") {
            // Redirect to login or other actions
        } else {
            alert(data.message);
        }
    });
});

// Similar functions for login and character creation




// Function to update the user's inventory UI
function updateInventory(inventory) {
    const inventoryDiv = document.getElementById("inventory");
    // Clear existing items
    inventoryDiv.innerHTML = "";
    // Add new items
    inventory.forEach(item => {
        const itemDiv = document.createElement("div");
        itemDiv.textContent = item;
        inventoryDiv.appendChild(itemDiv);
    });
}

// Function to update the room's items UI
function updateRoomItems(items) {
    const roomItemsDiv = document.getElementById("room-items");
    // Clear existing items
    roomItemsDiv.innerHTML = "";
    // Add new items
    items.forEach(item => {
        const itemDiv = document.createElement("div");
        itemDiv.textContent = item;
        roomItemsDiv.appendChild(itemDiv);
    });
}

document.addEventListener("DOMContentLoaded", () => {
    // Register form submission
    document.getElementById("register-form").addEventListener("submit", function(e) {
        // ... (Same as before)
    });

    // Login form submission
    document.getElementById("login-form").addEventListener("submit", function(e) {
        // ... (Same as before)
    });

    // Create character form submission
    document.getElementById("character-form").addEventListener("submit", function(e) {
        // ... (Same as before)
    });

    // Add to inventory
    function addToInventory(item) {
        // Call server route to add item to user's inventory
    }

    // Drop item into room
    function dropItem(item) {
        // Call server route to drop item into room
    }

    // Socket.IO for chat
    const socket = io.connect("http://localhost:5000");
    socket.on("connect", () => {
        socket.emit("join_room", {username: "John", room_name: "Room1"});
    });

    socket.on("message", (data) => {
        console.log("Received message:", data);
    });

    // Send chat message
    document.getElementById("send-message").addEventListener("click", function() {
        const message = document.getElementById("chat-input").value;
        socket.emit("send_message", {message: message, room_name: "Room1"});
    });
});