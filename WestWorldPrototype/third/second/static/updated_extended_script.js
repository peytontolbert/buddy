// Function to handle login form submission
document.getElementById('loginForm').addEventListener('submit', function(event) {
    event.preventDefault();
    const username = document.getElementById('username').value;
    // Simulate successful login
    onLoginSuccess();
});

// Function to handle character creation form submission
document.getElementById('characterForm').addEventListener('submit', function(event) {
    event.preventDefault();
    const attribute1 = document.getElementById('attribute1').value;
    // Simulate successful character creation
    onCharacterCreationSuccess();
});

// Called on successful login
function onLoginSuccess() {
    checkIfUserHasCharacter().then(hasCharacter => {
        if (hasCharacter) {
            loadMainRoomScreen();
        } else {
            loadCharacterCreationScreen();
        }
    });
}

// Called on successful character creation
function onCharacterCreationSuccess() {
    loadMainRoomScreen();
}

// Load character creation screen
function loadCharacterCreationScreen() {
    document.getElementById('characterCreationScreen').style.display = 'block';
    document.getElementById('loginScreen').style.display = 'none';
    document.getElementById('mainRoomScreen').style.display = 'none';
}

// Load main room screen
function loadMainRoomScreen() {
    document.getElementById('mainRoomScreen').style.display = 'block';
    document.getElementById('loginScreen').style.display = 'none';
    document.getElementById('characterCreationScreen').style.display = 'none';
}

// Function to check if user has a character (mockup)
function checkIfUserHasCharacter() {
    return new Promise((resolve, reject) => {
        setTimeout(() => {
            resolve(true);
        }, 1000);
    });
}
// Function to switch to registration screen
document.getElementById('switchToLogin').addEventListener('click', function() {
    document.getElementById('loginScreen').style.display = 'block';
    document.getElementById('registrationScreen').style.display = 'none';
});

// Function to handle registration form submission
document.getElementById('registrationForm').addEventListener('submit', function(event) {
    event.preventDefault();
    const username = document.getElementById('regUsername').value;
    const password = document.getElementById('regPassword').value;
    // Make API call to register the user
    // ...
});

// Function to send a chat message
function sendChatMessage(message) {
    // Make API call to send the message
    // ...
}

// Function to fetch and update chat history
function updateChatHistory() {
    // Make API call to get the last 50 messages
    // Update the chat UI
    // ...
}

// Function to enter the room
function enterRoom() {
    // Logic to enter the room and interact with the character
    // ...
}

// Function to load the character and room data
function loadCharacterAndRoom() {
    // Make API call to /api/get_character to get the character data
    // Based on the data, either provide an option to spawn in the room or stay in the chat area
    // ...
}
