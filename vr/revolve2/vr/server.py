import socket
import threading
import json


HOST = "127.0.0.1"
PORT = 5000


class Revolve2Server:
    def __init__(self, host, port):
        self.conn = None
        self.host = host
        self.port = port
        self.handler = None
        self.control_commands = []
        self.brains_data = []
        self.spliter = '<revolve2_spliter>'


    def send_mujoco_xml(self, mujoco_xml:str):
        if self.conn:
            # xml_data = {"type": "send_xml", "data": {"xml_data": mujoco_xml}}
            print(f"Sending mujoco xml: {mujoco_xml[:100]}")
            self.do_send_message(mujoco_xml.decode('utf8'))
            # self.conn.send(mujoco_xml)

    def do_send_message(self, message, conn = None):
        if not conn:
            conn = self.conn
        conn.sendall((message + self.spliter).encode('utf-8'))


    def handle_client(self, conn, addr):
        print(f"Connected by {addr}")
        self.conn = conn
        buffer = ""
        while True:
            try:
                data = conn.recv(655360).decode('utf-8')
                if not data:
                    break
                buffer += data
                while self.spliter in buffer:
                    message, buffer = buffer.split(self.spliter, 1)
                    if message:
                        self.process_message(json.loads(message), conn)

            except Exception as e:
                print(f"Error: {e}")
                break

        conn.close()
        print(f"Connection with {addr} closed.")

    def process_message(self, message, conn):
        if self.handler:
            self.handler(message)
        print(message)
        msg_type = message.get("type")
        msg_data = message.get("data")

        # 根据 type 执行不同逻辑
        if msg_type == "request_experiment":
            print("Received experiment request:", msg_data)

            if msg_data['msg'] == 'exp_1':
                self.run_experiment_1()
            elif msg_data['msg'] == 'exp_2':
                self.run_experiment_2()
            elif msg_data['msg'] == 'exp_3':
                self.run_experiment_3()

        elif msg_type == "send_experiment_data":
            print("Received experiment data:", msg_data)

        elif msg_type == "control_command":
            print("Control command received (unexpected at this step).")
            # self.send_control_commands()

        elif msg_type == "stop_server":
            print("Stopping server.")
            response = {"type": "stop_server", "success": True}
            self.do_send_message(json.dumps(response), conn)
        else:
            print("Unknown message type:", msg_type)


    def append_control_command(self, ctrl_index_position, position, ctrl_index_velocity, velocity):
        self.control_commands.append({"ctrl_index_position": ctrl_index_position, "position": position,
                         "ctrl_index_velocity": ctrl_index_velocity, "velocity": velocity
                                      })

    # send control command
    def send_control_commands(self):
        control_message = {
            "type": "control_command",
            "data": { "control": self.control_commands.copy() },
        }
        print(f"Sending control command: {control_message}")
        json_message = json.dumps(control_message)
        self.do_send_message(json_message)
        self.control_commands.clear()


    def append_brain_data(self, brain_data):
        self.brains_data.append(brain_data)


    def send_brains(self):
        brain_message = {
            "type": "brain",
            "data": {
                "brains": self.brains_data
            }
        }
        print(f"Sending brain message: {brain_message}")
        self.do_send_message(json.dumps(brain_message))
        self.brains_data.clear()

    def start_server(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            server.bind((HOST, PORT))
            server.listen()
            print(f"Server started at {HOST}:{PORT}")
            while True:
                conn, addr = server.accept()
                threading.Thread(target=self.handle_client, args=(conn, addr)).start()

    def run_experiment_1(self):
        from revolve2.vr.main import main
        main(self)
        # from revolve2.db.gen_rand_robots import main
        # main(self)

    def run_experiment_2(self):
        from revolve2.vr.examples.experiment_foundations.d_evaluate_multiple_interacting_robots.main import main
        main(self)

    def run_experiment_3(self, main=None):
        from revolve2.vr.examples.example_experiment_setups.e_robot_brain_cmaes.main import main
        main(self)

server = Revolve2Server(HOST, PORT)

if __name__ == "__main__":
    server.start_server()
