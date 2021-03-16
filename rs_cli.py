import sys
from typing import Optional, Tuple, List
from rs_hybrid import HybridRecommenderSystem
from rs_data import load_data

__version__ = "1.0.0"
__author__ = "wcrr51"


def print_info():
    print(f"Hybrid Recommender System V{__version__} by '{__author__}'")
    print(f"Dataset: Yelp Dataset [https://www.yelp.com/dataset/]'")
    print(f"Data Sample: Restaurants reviewed between 2015 and 2020")
    print(f"Score is a measure of how good a match the system thinks a restaurant is, "
          f"it is based on the underlying similarity of potential matches.")
    print(f"The RS looks at similar restaurants to those the user has previously "
          f"reviewed and returns those with the best similarity.")
    print(f"Stars (1.0-5.0) presented with recommendations represent predicted ratings by the system.")


def print_privacy_info():
    print(
        "None of the data input to the CLI is stored in permanent memory.\n"
        "The only user-based data used by the recommender system is review star ratings.\n"
        "First names are stored but not used in recommendations."
    )


def print_help():
    print("help\t- show available commands")
    print("userid\t- get the current user ID")
    print("userid <id>\t- set the selected user ID to <id>")
    print("recommendations <count>\t- get <count> recommendations for the current user")
    print("takeaway-only\t- get whether only restaurants offering takeaway or delivery are shown")
    print("takeaway-only <off/on>\t- set whether only restaurants offering takeaway or delivery are shown")
    print("info\t- show system information")
    print("privacy\t- show privacy and data usage information")
    print("exit\t- exit the program")


def input_str(message: Optional[str] = None) -> str:
    message = f"{message}\n > " if message is not None else " > "
    string = input(message)
    while string == "" or string.isspace():
        string = input(message)
    return string


def input_command() -> Tuple[str, List[str]]:
    command_parts = input_str().split()
    return command_parts[0], command_parts[1:]


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def isint(value):
    try:
        int(value)
        return True
    except ValueError:
        return False


def main() -> int:
    print("Loading data...")
    rs = HybridRecommenderSystem(*load_data())
    print("Done")

    # Output program and privacy information
    print()
    print_info()
    print()
    print_privacy_info()
    print()

    print("Type 'help' for a list of available commands.")
    user = rs.get_user_by_id("qmcTQ4RSOnKqW5eqc1CEfw")
    # location = None
    takeaway_only = True

    while True:
        # Prompt for command input
        command, args = input_command()

        # Check for valid command
        if command == "userid":
            if len(args) == 0:
                print(f"The current user ID is '{user.name}'" if user is not None else "No selected user.")
            elif len(args) == 1:
                user_id = args[0]
                if (u := rs.get_user_by_id(user_id)) is not None:
                    user = u
                    print(f"Set active user ID to '{user.name}' ({user['name']})")
                else:
                    print(f"Could not find user with ID '{user_id}'")
            else:
                print("Invalid number of arguments, usages:")
                print("userid")
                print("userid <id>")
        elif command == "takeaway_only":
            if len(args) == 0:
                print("Takeaway only is enabled." if takeaway_only else "Takeaway only is disabled.")
            elif len(args) == 1:
                if args[0] not in ("off", "on"):
                    print("Argument must be 'off' or 'on'")
                    continue
                if (args[0] == "on") ^ takeaway_only:
                    takeaway_only ^= True
                    print("Enabled takeaway only." if takeaway_only else "Disabled takeaway only.")
                else:
                    print("Takeaway only is already " + args[0].lower())
            else:
                print("Invalid number of arguments, usage:")
                print("takeaway_only <off/on>")
        elif command == "recommendations":
            # Validate user and argument value before giving recommendations
            if user is None:
                print("A user ID must be set for recommendations.")
                continue
            if len(args) != 1:
                print("Invalid number of arguments, usage:")
                print("recommendations <count>")
                continue
            count = 0
            if not isint(args[0]) or (count := int(args[0])) < 1:
                print("<count> must be a positive integer.")
                continue

            # Call to recommender system to get recommendations
            recommendations = rs.recommend_user(user.name, count)

            # Format recommendations into a user-friendly table
            recommendation_strings = [(
                str(i + 1),
                res["name"],
                f"{res['city']},", res["state"],
                f"{score:.2f}",
                f"{star_prediction:.2f}",
                "[✔] delivery or takeaway" if res["delivery_takeaway"] else "[✘] delivery or takeaway"
            ) for i, (res_id, _, score, star_prediction) in enumerate(recommendations.itertuples())
                if (res := rs.get_restaurant_by_id(res_id))["delivery_takeaway"]]
            string_lengths = [[len(part) for part in parts] for parts in recommendation_strings]
            s_p, s_n, s_c, s_st, s_si, s_sp, s_d = [max(string_lengths, key=lambda r: r[i])[i] for i in range(7)]
            print(f"Top {count} recommendations for '{user.name}' ({user['name']}):")
            for (pos, name, city, state, score, star_prediction, d_t) in recommendation_strings:
                print(
                    f"{pos:>{s_p}}: {name:{s_n}} [{city:{s_c}} {state:>{s_st}}] "
                    f"{score:{s_si}}-score, {star_prediction:{s_sp}}-stars, {d_t:{s_d}}"
                )
        elif command == "help":
            print_help()
        elif command == "info":
            print_info()
        elif command == "privacy":
            print_privacy_info()
        elif command == "exit":
            return 0
        else:
            print("Unrecognised command. Type 'help' for a list of commands.")


if __name__ == "__main__":
    sys.exit(main())
