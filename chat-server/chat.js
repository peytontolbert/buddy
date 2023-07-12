$(document).ready(function() {
    var refreshChatbox = function() {
        $.get('/chat', function(data) {
            $('#chatbox').empty();
            data.messages.forEach(function(message) {
                $('#chatbox').append('<p>' + message.user + ': ' + message.message + '</p>');
            });
        });
    };

    $('#chat-form').on('submit', function(event) {
        event.preventDefault();
        var message = $('#message').val();
        $.post('/chat', { 'user': 'username', 'message': message }, function() {
            $('#message').val('');
            refreshChatbox();
        });
    });

    refreshChatbox();
});