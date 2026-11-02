import React from 'react';
import { Button } from '@components/Button';
import { useUserData } from '@hooks/useUserData';

export interface UserCardProps {
  userId: string;
}

// Карточка пользователя с семантическими вызовами хуков и дочерних компонентов в JSX
export const UserCard: React.FC<UserCardProps> = ({ userId }) => {
  const { data } = useUserData(userId);

  const handleRefresh = () => {
    console.log('Обновление карточки', userId);
  };

  if (!data) {
    return <div>Загрузка...</div>;
  }

  return (
    <div className="user-card">
      <h3>{data.name}</h3>
      <p>{data.email}</p>
      <Button label="Обновить" onClick={handleRefresh} />
    </div>
  );
};
