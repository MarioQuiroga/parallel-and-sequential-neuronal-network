#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "server/Server.h"
#include "common/loaderMnist.h"

// Init Server
int main()
{
	Server s = Server(9000);
	return 0;
}