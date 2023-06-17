package python;

import plugin.api.API;
import rt4.client;

import java.io.*;
import java.net.ServerSocket;
import java.net.Socket;

public final class PythonSocket {

    static int port = 12346;

    static Thread thread;
    static boolean run = true;
    static private String data;
    static private ServerSocket serverSocket;


    public static void StartSocket(){

        Runnable runnable = () -> {
            try {
                serverSocket = new ServerSocket(port);
                while (run) {
                    Socket socket = serverSocket.accept();
                    System.out.println("New client Connected");

                    InputStream input = socket.getInputStream();
                    BufferedReader reader = new BufferedReader(new InputStreamReader(input));
                    //OutputStream output = socket.getOutputStream();
                    //PrintWriter writer = new PrintWriter(output, true);
                    String command = reader.readLine();
                    PythonCommands.ProcessCommand(command);
                    System.out.println(command);
                }

            } catch (IOException e) {
                e.printStackTrace();
            }

        };

        thread = new Thread(runnable);
        thread.start();

    }

    public static String GetData(){

        String ret = data;
        if (data != null) {
            data = null;
        }

        return ret;
    }

    public static void CloseSocket(){

        System.out.println("CLOSING SOCKET");

        run = false;

        try {
            serverSocket.close();

        } catch (Exception e) {

        }

    }


}
