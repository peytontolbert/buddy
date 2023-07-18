$(document).ready(function() {
    let username = '';

    $("#usernameButton").click(function() {
        username = $("#usernameInput").val();
        if (username) {
            // enable chat input and send button
            $("#userInput").prop('disabled', false);
            $("#sendButton").prop('disabled', false);
            // disable username input and button
            $("#usernameInput").prop('disabled', true);
            $("#usernameButton").prop('disabled', true);
        }
    });

    $("#sendButton").click(function() {
        var userInput = $("#userInput").val();

        if(userInput) {
            // Add user's message to chat
            appendMessage(username + ": " + userInput);

            // Send the user's message to the server and get the response
            fetch('http://localhost:5000/nuggetchat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ user: username, message: userInput }),
            })
            .then(response => response.json())
            .then(data => {
                // Add the bot's response to chat
                appendMessage("Nugget: " + data.response);
                // Clear the input field
                $("#userInput").val("");
            })
            .catch((error) => {
                console.error('Error:', error);
            });
        }
    });
});


const appendMessage = (messsage) => {
    $("#chatBox").append("<p> " + messsage + "</p>");
}