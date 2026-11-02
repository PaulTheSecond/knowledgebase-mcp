export interface UserDto {
  id: string;
  name: string;
  email: string;
}

// Сервис обращения к бэкенду пользователей
export async function fetchUserApi(id: string): Promise<UserDto> {
  return {
    id,
    name: 'Иван Иванов',
    email: 'ivan@example.com'
  };
}
