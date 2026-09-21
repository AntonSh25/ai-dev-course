import os
from time import perf_counter

import openai
from dotenv import load_dotenv
from openai import OpenAI


BASE_URL = "https://api.proxyapi.ru/openai/v1"
MODEL = "gpt-4.1"


def summarize_request(client: OpenAI, user_text: str) -> None:
    """Кратко пересказывает обращение и печатает метрики запроса."""
    text = user_text.strip()
    if not text:
        print("Ошибка: обращение не должно быть пустым.")
        return

    started_at = perf_counter()

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Сформулируй краткое содержание обращения в одном "
                        "предложении. Не добавляй факты, которых нет в тексте.\n\n"
                        f"Обращение: {text}"
                    ),
                }
            ],
        )
    except openai.AuthenticationError:
        print("Ошибка авторизации: проверьте LLM_API_KEY.")
        return
    except openai.PermissionDeniedError:
        print("Нет доступа к модели: проверьте разрешения API-ключа.")
        return
    except openai.RateLimitError:
        print("Достигнут лимит запросов провайдера. Повторите запрос позднее.")
        return
    except openai.APITimeoutError:
        print("ProxyAPI не успел ответить за отведенное время.")
        return
    except openai.APIConnectionError:
        print("Не удалось соединиться с ProxyAPI. Проверьте сеть.")
        return
    except openai.APIStatusError as error:
        if error.status_code == 402:
            print("Недостаточно средств на балансе ProxyAPI.")
        else:
            print(f"API вернул ошибку со статусом {error.status_code}.")
        if error.request_id:
            print(f"Request ID: {error.request_id}")
        return

    elapsed_seconds = perf_counter() - started_at
    choice = response.choices[0]
    answer = choice.message.content

    print("\nРезультат:")
    print(answer or "Модель не вернула текстовый ответ")

    print("\nМетрики:")
    print(f"Модель: {response.model}")
    print(f"Завершение: {choice.finish_reason}")
    print(f"Время: {elapsed_seconds:.2f} с")

    if response.usage is not None:
        print(f"Входные токены: {response.usage.prompt_tokens}")
        print(f"Выходные токены: {response.usage.completion_tokens}")
        print(f"Всего токенов: {response.usage.total_tokens}")

    print(f"ID ответа: {response.id}")


def main() -> None:
    load_dotenv()

    token = os.getenv("LLM_API_KEY")
    if not token:
        raise SystemExit(
            "Не найдена переменная LLM_API_KEY. "
            "Проверьте файл .env в корне проекта."
        )

    client = OpenAI(base_url=BASE_URL, api_key=token)
    user_text = input("Введите текст обращения: ")
    summarize_request(client, user_text)


if __name__ == "__main__":
    main()