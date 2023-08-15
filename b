<!doctype html>

<div id="popup" class="popup">
  <div class="popup-content">
    <span class="popup-close">&times;</span>
    <p>Welcome to my website!</p>
  </div>
</div>

/* Popup styling */
.popup {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-color: rgba(0, 0, 0, 0.5);
  display: flex;
  justify-content: center;
  align-items: center;
}

.popup-content {
  background-color: white;
  padding: 20px;
  border-radius: 10px;
  text-align: center;
}

.popup-close {
  position: absolute;
  top: 10px;
  right: 10px;
  cursor: pointer;
}

/* Hide the popup by default */
.popup {
  display: none;
}

// Show the popup
document.getElementById("popup").style.display = "flex";

// Close the popup
var close = document.getElementsByClassName("popup-close")[0];
close.onclick = function() {
  document.getElementById("popup").style.display = "none";
}
