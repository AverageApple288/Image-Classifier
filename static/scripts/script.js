function trainButtonClicked() {
    console.log("Training started");
    var trainButton = document.getElementById("trainButton");
    var enabled = trainButton.style.display;
    if (enabled === "none") {
        trainButton.style.display = "block";
    }
}
