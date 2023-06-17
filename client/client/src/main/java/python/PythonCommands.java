package python;

import org.openrs2.deob.annotation.Pc;
import plugin.api.API;
import rt4.*;

public class PythonCommands {

    public static void ProcessCommand(String command){
        String[] split = command.split(" ");

        for (int i=0; i< split.length;i++){
            split[i] = split[i].trim();
        }

        command = split[0];


        switch(command){
            case "login":
                //new AutoLogin().Login(split[1], client);
                break;
        }
    }


}
