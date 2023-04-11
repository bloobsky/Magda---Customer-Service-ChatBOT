
const msgerForm = get(".msger-inputarea");
const msgerInput = get(".msger-input");
const msgerChat = get(".msger-chat");

const BOT_IMG = "/static/bot_icon_2.png";
const PERSON_IMG = "/static/human_icon.png";
const BOT_NAME = "Magda";
const PERSON_NAME = "You";

// Load chat history from local storage on page load
if (localStorage.getItem("chatHistory")) {
    const chatHistory = JSON.parse(localStorage.getItem("chatHistory"));
    chatHistory.forEach((msg) => {
        appendMessage(msg.name, msg.img, msg.side, msg.text, new Date(msg.time));
    });
}

msgerForm.addEventListener("submit", (event) => {
    event.preventDefault();

    const msgText = msgerInput.value;
    if (!msgText) return;

    appendMessage(PERSON_NAME, PERSON_IMG, "right", msgText);

    // Save message to local storage
    const chatHistory = JSON.parse(localStorage.getItem("chatHistory")) || [];
    chatHistory.push({
        name: PERSON_NAME,
        img: PERSON_IMG,
        side: "right",
        text: msgText,
        time: new Date().getTime(),
    });
    localStorage.setItem("chatHistory", JSON.stringify(chatHistory));

    msgerInput.value = "";
    botResponse(msgText);
});

function appendMessage(name, img, side, text, date = new Date()) {
    //   Simple solution for small apps
    const msgHTML = `
<div class="msg ${side}-msg">
    <div class="msg-img" style="background-image: url(${img})"></div>

    <div class="msg-bubble">
    <div class="msg-info">
        <div class="msg-info-name">${name}</div>
        <div class="msg-info-time">${formatDate(date)}</div>
    </div>

    <div class="msg-text">${text}</div>
    </div>
</div>
`;

    msgerChat.insertAdjacentHTML("beforeend", msgHTML);
    msgerChat.scrollTop += 500;
}

function botResponse(rawText) {
    // Bot Response
    $.get("/get", { msg: rawText }).done(function (data) {
        console.log(rawText);
        console.log(data);
        const msgText = data;
        appendMessage(BOT_NAME, BOT_IMG, "left", msgText);

        // Save message to local storage
        const chatHistory = JSON.parse(localStorage.getItem("chatHistory")) || [];
        chatHistory.push({
            name: BOT_NAME,
            img: BOT_IMG,
            side: "left",
            text: msgText,
            time: new Date().getTime(),
        });
        localStorage.setItem("chatHistory", JSON.stringify(chatHistory));
    });
}

// Utils
function get(selector, root = document) {
    return root.querySelector(selector);
}

function formatDate(date) {
    const h = "0" + date.getHours();
    const m = "0" + date.getMinutes();

    return `${h.slice(-2)}:${m.slice(-2)}`;
}
function clearChatHistory() {
    localStorage.removeItem("chatHistory");
    location.reload();
}

