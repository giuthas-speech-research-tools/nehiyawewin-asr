from datasets import load_dataset, DatasetDict


def main():
    print("Hello from whisper-test!")
    common_voice = DatasetDict()
    common_voice["train"] = load_dataset(
        path="cree-asr/hindi",
        name="hi",
        split="train+validation",
        use_auth_token=True
    )
    common_voice["test"] = load_dataset(
        path="cree-asr/hindi",
        name="default",
        split="test",
        use_auth_token=True
    )
    print(common_voice)


if __name__ == "__main__":
    main()
