void test(char *str)
{
	char buf[40];
	if(strlen(str) > 40)
		return;
	strcpy(buf, str);	/* FLAW */
	printf("result: %s\n", buf);
}

int main(int argc, char **argv)
{
	char *userstr;
	if(argc > 1) {
		userstr = argv[1];
		test(userstr);
		return 1;
	}
	return 0;
}
