package main

import (
	"fmt"
	"log"

	"example.com/greetings"
)

func main() {
	log.SetPrefix("Greetings:")
	log.SetFlags(0)
	// message, err := greetings.Hello("Tong")
	names := []string{"Gladys", "Samantha", "Darrin"}

	messages, err := greetings.Hellos(names)

	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(messages)

}
