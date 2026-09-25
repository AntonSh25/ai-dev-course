from time import perf_counter

import openai
from openai import OpenAI
from pydantic import ValidationError

from config import Settings

SYSTEM_INSTRUCTION = (
    "Ты помогаешь оператору службы поддержки. "
    "Кратко пересказывай обращение одним предложением. "
    "Используй только факты из обращения. "
    "Не придумывай суммы, даты, причины и действия. "
    "Если данных недостаточно, прямо сообщай об этом."
)

def print_usage(usage) -> None:
    if usage is None:
        print("Провайдер не вернул статистику токенов")
        return
    print(f"Входные токены: {usage.prompt_tokens}")
    print(f"Выходные токены: {usage.completion_tokens}")
    print(f"Всего токенов: {usage.total_tokens}")
    
def build_messages(user_text: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": SYSTEM_INSTRUCTION,
        },
        {
            "role": "user",
            "content": f"Обращение:\n{user_text}",
        },
    ]

def validate_summary(summary: str | None) -> str:
    if summary is None:
        raise ValueError("Модель не вернула текст")

    cleaned = summary.strip()
    if not cleaned:
        raise ValueError("Модель вернула пустой ответ")
    if len(cleaned) > 300:
        raise ValueError("Резюме получилось слишком длинным")

    return cleaned

def summarize_request(client: OpenAI, user_text: str, settings: Settings) -> None:
    """Кратко пересказывает обращение и печатает метрики запроса."""
    text = user_text.strip()
    if not text:
        print("Ошибка: обращение не должно быть пустым.")
        return

    started_at = perf_counter()

    try:
        response = client.chat.completions.create(
            model=settings.model,
            messages=build_messages(text),
            temperature=settings.temperature,
            max_completion_tokens=settings.max_output_tokens,
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
    if choice.finish_reason == "length":
        print(
            "Ответ модели остановлен из-за ограничения длины: "
            f"{settings.max_output_tokens} токенов."
        )
        return

    if choice.finish_reason != "stop":
        print(
            "Модель не вернула готовое резюме. "
            f"Причина завершения: {choice.finish_reason}."
        )
        return
    
    try:
        answer = validate_summary(choice.message.content)
    except ValueError as error:
        print(f"Некорректный ответ модели: {error}")
        return

    print("\nРезультат:")
    print(answer)

    print("\nМетрики:")
    print(f"Модель: {response.model}")
    print(f"Завершение: {choice.finish_reason}")
    print(f"Время: {elapsed_seconds:.2f} с")

    print_usage(response.usage)

    print(f"ID ответа: {response.id}")


def main() -> None:
    try:
        settings = Settings()
    except ValidationError as error:
        print("Ошибка конфигурации:")
        for issue in error.errors():
            field = ".".join(str(part) for part in issue["loc"])
            print(f"- {field}: {issue['msg']}")
        return

    client = OpenAI(
        base_url=str(settings.base_url),
        api_key=settings.llm_api_key.get_secret_value(),
    )

    print(f"Окружение: {settings.app_env}")
    print(f"Модель: {settings.model}")

    user_text = input("Введите текст обращения: ")
    summarize_request(client, user_text, settings)

if __name__ == "__main__":
    main()