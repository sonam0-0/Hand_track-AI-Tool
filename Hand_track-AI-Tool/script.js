document.addEventListener("DOMContentLoaded", function () {
    const button = document.querySelector(".try-now");

    button.addEventListener("click", function () {
        alert("Redirecting to Sign-Up Section!");
        document.getElementById("signup").scrollIntoView({ behavior: "smooth" });
    });
});
