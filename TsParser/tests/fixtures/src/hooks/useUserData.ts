import { useState, useEffect } from 'react';
import { fetchUserApi, UserDto } from '@services/api';

// Пользовательский хук получения данных профиля
export function useUserData(userId: string) {
  const [data, setData] = useState<UserDto | null>(null);

  useEffect(() => {
    fetchUserApi(userId).then(setData);
  }, [userId]);

  return { data };
}
