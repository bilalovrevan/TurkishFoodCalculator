let isRegister=false;

function openAuth(reg=false){
    isRegister=reg;
    document.getElementById("authTitle").innerText=reg?"Register":"Login";
    document.getElementById("authModal").style.display="flex";
}

function toggleAuth(){
    isRegister=!isRegister;
    document.getElementById("authTitle").innerText=isRegister?"Register":"Login";
}

function submitAuth(){
    const email=document.getElementById("email").value;
    const pass=document.getElementById("password").value;
    if(!email||!pass) return alert("Fill all fields");

    const users=JSON.parse(localStorage.getItem("users")||"{}");

    if(isRegister){
        if(users[email]) return alert("User exists");
        users[email]={password:pass,history:null};
    }else{
        if(!users[email]||users[email].password!==pass)
            return alert("Invalid login");
    }

    localStorage.setItem("users",JSON.stringify(users));
    localStorage.setItem("currentUser",email);
    document.getElementById("authModal").style.display="none";
    alert("Logged in as "+email);
}
