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