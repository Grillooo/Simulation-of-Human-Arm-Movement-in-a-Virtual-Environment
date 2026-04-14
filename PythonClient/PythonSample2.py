# PythonSample2.py
# Filtered OptiTrack NatNet client output.
# Only prints frames where markers are detected, showing:
#   - Frame ID
#   - Marker ID and position for each labeled marker
#
# Based on PythonSample.py but with all verbose output suppressed.

import sys
import time
import socket
from NatNetClient import NatNetClient

# --- UDP Configuration (must match UDPMarkerReceiver.listenPort in Unity) ---
UNITY_IP = "127.0.0.1"
UNITY_PORT = 5005

# Global UDP socket
_udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def receive_new_frame(data_dict):
    """Callback for each mocap frame. Only prints when labeled markers exist."""
    frame_number = data_dict.get("frameNumber", "?")
    labeled_count = data_dict.get("labeledMarkerCount", 0)

    # Skip frames with no detected markers
    if labeled_count == 0:
        return

    print(f"Frame {frame_number}  |  Labeled Markers: {labeled_count}")


def receive_new_frame_with_data(data_dict):
    """
    Callback with full mocap data. Filters to only show frames where
    cameras detect markers, displaying Frame ID and Marker IDs.
    Also forwards marker data to Unity via UDP.
    """
    frame_number = data_dict.get("frameNumber", "?")
    mocap_data = data_dict.get("mocap_data", None)

    if mocap_data is None:
        return

    labeled_marker_data = mocap_data.labeled_marker_data
    if labeled_marker_data is None:
        return

    marker_list = labeled_marker_data.labeled_marker_list
    if len(marker_list) == 0:
        return

    # Print frame header
    print(f"Frame {frame_number}")

    # Build UDP message: frame_id;marker_count;id,x,y,z;id,x,y,z;...
    udp_parts = [str(frame_number), str(len(marker_list))]

    for marker in marker_list:
        # Decode the compound marker ID into model_id and marker_id
        model_id = marker.id_num >> 16
        marker_id = marker.id_num & 0x0000FFFF
        pos = marker.pos

        print(f"  Marker {marker_id:5d}  |  pos: [{pos[0]:7.4f}, {pos[1]:7.4f}, {pos[2]:7.4f}]")
        udp_parts.append(f"{marker_id},{pos[0]:.6f},{pos[1]:.6f},{pos[2]:.6f}")

    # Send to Unity
    message = ";".join(udp_parts)
    _udp_socket.sendto(message.encode("utf-8"), (UNITY_IP, UNITY_PORT))

    print()  # Blank line between frames


def receive_rigid_body_frame(new_id, position, rotation):
    pass


def print_configuration(natnet_client):
    natnet_client.refresh_configuration()
    print("Connection Configuration:")
    print("  Client:          %s" % natnet_client.local_ip_address)
    print("  Server:          %s" % natnet_client.server_ip_address)
    print("  Command Port:    %d" % natnet_client.command_port)
    print("  Data Port:       %d" % natnet_client.data_port)

    if natnet_client.use_multicast:
        print("  Using Multicast")
        print("  Multicast Group: %s" % natnet_client.multicast_address)
    else:
        print("  Using Unicast")

    application_name = natnet_client.get_application_name()
    server_version = natnet_client.get_server_version()
    print("  NatNet Server Info")
    print("    Application Name %s" % application_name)
    print("    MotiveVersion  %d.%d.%d.%d" % (
        server_version[0], server_version[1],
        server_version[2], server_version[3]))
    print("  PythonVersion    %s" % sys.version)


def my_parse_args(arg_list, args_dict):
    arg_list_len = len(arg_list)
    if arg_list_len > 1:
        args_dict["serverAddress"] = arg_list[1]
        if arg_list_len > 2:
            args_dict["clientAddress"] = arg_list[2]
        if arg_list_len > 3:
            if len(arg_list[3]):
                args_dict["use_multicast"] = True
                if arg_list[3][0].upper() == "U":
                    args_dict["use_multicast"] = False
        if arg_list_len > 4:
            args_dict["stream_type"] = arg_list[4]
    return args_dict


if __name__ == "__main__":

    optionsDict = {}
    optionsDict["clientAddress"] = "127.0.0.1"
    optionsDict["serverAddress"] = "127.0.0.1"
    optionsDict["use_multicast"] = None
    optionsDict["stream_type"] = None

    optionsDict = my_parse_args(sys.argv, optionsDict)
    streaming_client = NatNetClient()
    streaming_client.set_client_address(optionsDict["clientAddress"])
    streaming_client.set_server_address(optionsDict["serverAddress"])

    # Use the data-enriched callback so we can access labeled marker details
    streaming_client.new_frame_with_data_listener = receive_new_frame_with_data
    streaming_client.rigid_body_listener = receive_rigid_body_frame

    # Suppress the NatNetClient's own verbose frame dump
    streaming_client.set_print_level(0)

    print("=" * 60)
    print("  OptiTrack Filtered Output")
    print("  Showing only frames with detected markers")
    print("  (Frame ID + Marker IDs)")
    print(f"  UDP forwarding to Unity: {UNITY_IP}:{UNITY_PORT}")
    print("=" * 60)
    print()

    # Select Multicast or Unicast
    cast_choice = input("Select 0 for multicast and 1 for unicast: ")
    cast_choice = int(cast_choice)
    while cast_choice != 0 and cast_choice != 1:
        cast_choice = input("Invalid option. Select 0 for multicast or 1 for unicast: ")
        cast_choice = int(cast_choice)

    if cast_choice == 0:
        optionsDict["use_multicast"] = True
    else:
        optionsDict["use_multicast"] = False
    streaming_client.set_use_multicast(optionsDict["use_multicast"])

    # Client / Server addresses
    client_addr_choice = input("Client Address (127.0.0.1): ")
    if client_addr_choice != "":
        streaming_client.set_client_address(client_addr_choice)

    server_addr_choice = input("Server Address (127.0.0.1): ")
    if server_addr_choice != "":
        streaming_client.set_server_address(server_addr_choice)

    # Select data or command stream
    stream_choice = None
    while stream_choice != 'd' and stream_choice != 'c':
        stream_choice = input("Select d for datastream and c for command stream: ")
    optionsDict["stream_type"] = stream_choice

    # Start streaming client
    is_running = streaming_client.run(optionsDict["stream_type"])
    if not is_running:
        print("ERROR: Could not start streaming client.")
        try:
            sys.exit(1)
        except SystemExit:
            print("...")
        finally:
            print("exiting")

    time.sleep(1)
    if streaming_client.connected() is False:
        print("ERROR: Could not connect properly. Check that Motive streaming is on.")
        try:
            sys.exit(2)
        except SystemExit:
            print("...")
        finally:
            print("exiting")

    print_configuration(streaming_client)
    print()
    print("Streaming... (press 'q' + Enter to quit)")
    print()

    is_looping = True
    while is_looping:
        inchars = input('')
        if len(inchars) > 0:
            c1 = inchars[0].lower()
            if c1 == 'q':
                is_looping = False
                streaming_client.shutdown()
                break

    _udp_socket.close()
    print("exiting")
