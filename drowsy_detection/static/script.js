document.getElementById("startDemo").addEventListener("click", function() {
    fetch("/start-demo")
    .then(response => response.json())
    .then(data => {
        alert(data.message);
    })
    .catch(error => {
        console.error("Error starting demo:", error);
    });
});
function startDetection() {
    fetch('/start-demo')
        .then(response => response.json())
        .then(data => {
            alert(data.message); // Show success or error message
        })
        .catch(error => {
            console.error('Error:', error);
            alert("Failed to start demo.");
        });
}
