package python;

import rt4.JagString;
import rt4.LoginManager;
import rt4.client;

public class AutoLogin {


    public static void Login()
    {

        String username = "Temp";
        if (Python.username != null){
            username = Python.username;
        }

        JagString usernameInput = JagString.parse(username);
        LoginManager.method3896(usernameInput, usernameInput, 0);
    }
}
