/**********************************
tcp_ser.c: the source file of the server in TCP transmission
***********************************/
#include "headsock.h"
#define BACKLOG 10   // Maximum number of pending client connections
// Function declaration: handle data receiving from the client
void str_ser(int sockfd);
int main(void)
{
    int sockfd, con_fd, ret;
    struct sockaddr_in my_addr;      // Server address structure
    struct sockaddr_in their_addr;   // Client address structure
    int sin_size;
    pid_t pid;

    // Create a TCP socket
    // AF_INET means IPv4, SOCK_STREAM means TCP
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0)
    {
        printf("error in socket!");
        exit(1);
    }
    // Set server address information
    my_addr.sin_family = AF_INET;                 // Use IPv4 address family
    my_addr.sin_port = htons(MYTCP_PORT);         // Set server port number
    my_addr.sin_addr.s_addr = htonl(INADDR_ANY);  // Accept connections from any IP address
    bzero(&(my_addr.sin_zero), 8);                // Clear the unused bytes in the structure
    // Bind the socket to the specified IP address and port
    ret = bind(sockfd, (struct sockaddr *) &my_addr, sizeof(struct sockaddr));
    if (ret < 0)
    {
        printf("error in binding");
        exit(1);
    }
    // Listen for incoming client connections
    ret = listen(sockfd, BACKLOG);
    if (ret < 0)
    {
        printf("error in listening");
        exit(1);
    }
    printf("receiving start\n");
    // Keep the server running to accept multiple client connections
    while (1)
    {
        sin_size = sizeof(struct sockaddr_in);

        // Accept a connection request from a client
        // con_fd is a new socket used to communicate with this client
        con_fd = accept(sockfd, (struct sockaddr *)&their_addr, &sin_size);
        if (con_fd < 0)
        {
            printf("error in accept\n");
            exit(1);
        }
        // Create a child process to handle the connected client
        if ((pid = fork()) == 0)
        {
            // Child process does not need the listening socket
            close(sockfd);
            // Receive and process data from the client
            str_ser(con_fd);
            // Close the connected socket after communication
            close(con_fd);
            // End the child process
            exit(0);
        }
        else
        {
            // Parent process closes the connected socket
            // and continues waiting for new clients
            close(con_fd);
        }
    }

    // Close the listening socket
    close(sockfd);
    exit(0);
}
// Function used to receive data from the client
void str_ser(int sockfd)
{
    char recvs[MAXSIZE];  // Buffer used to store received data
    int n = 0;
    // Receive data from the client through the connected socket
    // MAXSIZE defines the maximum number of bytes to receive
    if ((n = recv(sockfd, &recvs, MAXSIZE, 0)) == -1)
    {
        printf("receiving error!\n");
        return;
    }
    // Add string terminator to make the received data printable
    recvs[n] = '\0';
    // Print the received string
    printf("the received string:\n%s\n", recvs);
}
