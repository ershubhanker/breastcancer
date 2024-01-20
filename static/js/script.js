document.addEventListener('DOMContentLoaded', function () {
    var hamburger = document.querySelector('.hamburger-menu');
    var sidebar = document.getElementById('sidebar');
    var content = document.getElementById('upload-content'); // Make sure this ID is correct

    hamburger.addEventListener('click', function () {
        sidebar.classList.toggle('minimized');
        content.classList.toggle('content-expanded'); // Add this line
    });
});

